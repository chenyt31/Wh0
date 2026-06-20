import os
import time
import argparse
import threading
import numpy as np
import json
import base64
import cv2
import asyncio
import websockets
from io import BytesIO
from scipy.spatial.transform import Rotation as R
from PIL import Image

from paths import WH0_ROOT, configure_imports

configure_imports(require_xr=True)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from teleop.robot_control.robot_arm import (
    G1_29_ArmController, G1_23_ArmController,
    H1_2_ArmController, H1_ArmController
)
from teleop.robot_control.robot_arm_ik import (
    G1_29_ArmIK, G1_23_ArmIK,
    H1_2_ArmIK, H1_ArmIK
)
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from teleimager.image_client import ImageClient
from vitra.datasets.coord_trans import get_camera_pose_in_base_frame
import pinocchio as pin

import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

# For Inspire hand direct control
from unitree_sdk2py.core.channel import ChannelPublisher
from inspire_sdkpy import inspire_dds
import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default

INSPIRE_LEFT_COMMAND_TOPIC = "rt/inspire_hand/ctrl/l"
INSPIRE_RIGHT_COMMAND_TOPIC = "rt/inspire_hand/ctrl/r"


# Lower-body lock targets from the Unitree replay stack.
G1_29_LOWER_BODY_LOCKED_Q = {
    0: -0.2845, 1: -0.0014, 2: -0.0036, 3: 0.6037, 4: -0.3828, 5: 0.0039,
    6: -0.2730, 7: -0.0035, 8: 0.0098, 9: 0.5994, 10: -0.3893, 11: -0.0003,
    12: 0.0062, 13: 0.0000, 14: 0.0000,
}
G1_23_LOWER_BODY_LOCKED_Q = {
    0: -0.2845, 1: -0.0014, 2: -0.0036, 3: 0.6037, 4: -0.3828, 5: 0.0039,
    6: -0.2730, 7: -0.0035, 8: 0.0098, 9: 0.5994, 10: -0.3893, 11: -0.0003,
    12: 0.0062, 13: 0.0000, 14: 0.0000,
}
H1_2_LOWER_BODY_LOCKED_Q = {i: 0.0 for i in range(13)}
H1_LOWER_BODY_LOCKED_Q = {i: 0.0 for i in range(12)}


class LockedArmController:
    """Arm controller that locks lower body joints."""
    def __init__(self, arm_controller, lower_body_locked_q, simulation_mode=False):
        self.base_ctrl = arm_controller
        self.lower_body_locked_q = lower_body_locked_q
        self.simulation_mode = simulation_mode
        self.ctrl_lock = threading.Lock()
        logger_mp.info("[LockedArmController] Initialized.")

    def ctrl_dual_arm(self, arm_q_target, arm_tauff_target):
        with self.ctrl_lock:
            self.base_ctrl.ctrl_dual_arm(arm_q_target, arm_tauff_target)

    def get_current_dual_arm_q(self):
        return self.base_ctrl.get_current_dual_arm_q()

    def get_current_dual_arm_dq(self):
        return self.base_ctrl.get_current_dual_arm_dq()

    def ctrl_dual_arm_go_home(self):
        self.base_ctrl.ctrl_dual_arm_go_home()


class InspireHandDirectController:
    """Direct controller for Inspire hand."""
    def __init__(self, simulation_mode=False):
        self.simulation_mode = simulation_mode
        if not self.simulation_mode:
            self.left_hand_publisher = ChannelPublisher(
                INSPIRE_LEFT_COMMAND_TOPIC, inspire_dds.inspire_hand_ctrl
            )
            self.left_hand_publisher.Init()
            self.right_hand_publisher = ChannelPublisher(
                INSPIRE_RIGHT_COMMAND_TOPIC, inspire_dds.inspire_hand_ctrl
            )
            self.right_hand_publisher.Init()
            logger_mp.info("[InspireHandDirectController] Publishers initialized.")
        else:
            logger_mp.info("[InspireHandDirectController] Simulation mode.")

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        if self.simulation_mode:
            return

        if len(left_q_target) < 6:
            left_q_target = list(left_q_target) + [0.0] * (6 - len(left_q_target))
        if len(right_q_target) < 6:
            right_q_target = list(right_q_target) + [0.0] * (6 - len(right_q_target))

        scaled_left = [int(np.clip(val * 1000, 0, 1000)) for val in left_q_target[:6]]
        scaled_right = [int(np.clip(val * 1000, 0, 1000)) for val in right_q_target[:6]]

        left_cmd = inspire_hand_default.get_inspire_hand_ctrl()
        left_cmd.angle_set = scaled_left
        left_cmd.mode = 0b0001
        self.left_hand_publisher.Write(left_cmd)

        right_cmd = inspire_hand_default.get_inspire_hand_ctrl()
        right_cmd.angle_set = scaled_right
        right_cmd.mode = 0b0001
        self.right_hand_publisher.Write(right_cmd)


def image_to_base64(image: np.ndarray, fmt="JPEG"):
    """Convert OpenCV image (BGR) to base64 string."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    pil = Image.fromarray(image_rgb)
    buf = BytesIO()
    pil.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def compute_state_in_camera_frame(arm_ik, arm_q, left_hand_qpos, right_hand_qpos, waist_yaw=0.0, waist_roll=0.0, waist_pitch=0.0):
    """
    Compute robot state in camera coordinate frame.

    Args:
        arm_ik: ArmIK instance (G1_29_ArmIK, etc.)
        arm_q: Current arm joint angles (numpy array)
        left_hand_qpos: Left hand joint positions (6-dim list/array)
        right_hand_qpos: Right hand joint positions (6-dim list/array)
        waist_yaw: Waist yaw angle (rad)
        waist_roll: Waist roll angle (rad)
        waist_pitch: Waist pitch angle (rad)

    Returns:
        state: 24-dim numpy array [left_trans(3) + left_euler(3) + left_hand(6) + right_trans(3) + right_euler(3) + right_hand(6)]
    """
    # 1. Compute forward kinematics to get wrist poses in base frame
    pin.forwardKinematics(arm_ik.reduced_robot.model, arm_ik.reduced_robot.data, arm_q)

    # Get wrist poses in base frame (updateFramePlacement automatically updates frame placements)
    left_wrist_base = pin.updateFramePlacement(
        arm_ik.reduced_robot.model,
        arm_ik.reduced_robot.data,
        arm_ik.L_hand_id
    ).homogeneous

    right_wrist_base = pin.updateFramePlacement(
        arm_ik.reduced_robot.model,
        arm_ik.reduced_robot.data,
        arm_ik.R_hand_id
    ).homogeneous

    # 2. Get camera pose in base frame
    T_base_camera = get_camera_pose_in_base_frame(waist_yaw, waist_roll, waist_pitch)
    T_camera_base = np.linalg.inv(T_base_camera)

    # 3. Transform wrist poses to camera frame
    # left_wrist_camera = T_camera_base @ left_wrist_base
    # right_wrist_camera = T_camera_base @ right_wrist_base

    # 4. Extract translation and euler angles
    left_trans = left_wrist_base[:3, 3]
    left_rot = R.from_matrix(left_wrist_base[:3, :3])
    left_euler = left_rot.as_euler('xyz')  # or 'zyx', check convention

    right_trans = right_wrist_base[:3, 3]
    right_rot = R.from_matrix(right_wrist_base[:3, :3])
    right_euler = right_rot.as_euler('xyz')  # or 'zyx', check convention

    # 5. Ensure hand qpos are 6-dim
    left_hand = np.array(left_hand_qpos[:6]) if len(left_hand_qpos) >= 6 else np.pad(left_hand_qpos, (0, 6-len(left_hand_qpos)), 'constant')
    right_hand = np.array(right_hand_qpos[:6]) if len(right_hand_qpos) >= 6 else np.pad(right_hand_qpos, (0, 6-len(right_hand_qpos)), 'constant')

    # 6. Concatenate: [left_trans(3) + left_euler(3) + left_hand(6) + right_trans(3) + right_euler(3) + right_hand(6)]
    state = np.concatenate([
        left_trans,      # 3
        left_euler,      # 3
        left_hand,       # 6
        right_trans,     # 3
        right_euler,     # 3
        right_hand       # 6
    ])

    return state


class VLAClient:
    def __init__(self, server_ip, server_port=8765):
        self.server_ip = server_ip
        self.server_port = server_port
        self.websocket = None
        self.uri = f"ws://{server_ip}:{server_port}"
        self.loop = None
        self._loop_thread = None

    def _run_async_loop(self):
        """Run asyncio event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def connect(self):
        """Connect to WebSocket server."""
        try:
            # Start event loop in a separate thread if not already running
            if self.loop is None:
                self._loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
                self._loop_thread.start()
                # Wait a bit for loop to start
                time.sleep(0.2)
            elif not self.loop.is_running():
                # Loop exists but not running, restart it
                self._loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
                self._loop_thread.start()
                time.sleep(0.2)

            # Connect synchronously using run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(
                self._async_connect(),
                self.loop
            )
            result = future.result(timeout=5.0)
            if result:
                logger_mp.info(f"Connected to VLA Server at {self.uri}")
            return result
        except Exception as e:
            logger_mp.error(f"Failed to connect to VLA Server: {e}")
            import traceback
            logger_mp.debug(traceback.format_exc())
            return False

    async def _async_connect(self):
        """Async connection helper."""
        try:
            self.websocket = await websockets.connect(self.uri)
            return True
        except Exception as e:
            logger_mp.error(f"WebSocket connection error: {e}")
            return False

    def send_state(self, image, instruction, state, left_active=False, right_active=True, prediction_horizon=16):
        """
        Send state to server.
        image: OpenCV image (BGR)
        instruction: str
        state: list or dict
        left_active: bool, whether left hand is active
        right_active: bool, whether right hand is active
        prediction_horizon: int, number of future steps to predict (default 16)
        """
        if not self.websocket or not self.loop:
            return

        try:
            # Convert image to base64
            img_b64 = image_to_base64(image)

            payload = {
                "type": "inference_request",
                "data": {
                    "image": img_b64,
                    "instruction": instruction,
                    "state": state if isinstance(state, list) else state.tolist() if hasattr(state, 'tolist') else list(state),
                    "state_mask": [left_active, right_active],
                    "action_mask": [[left_active, right_active]] * prediction_horizon,
                    "timestamp": time.time()
                }
            }

            # Send asynchronously
            future = asyncio.run_coroutine_threadsafe(
                self._async_send(payload),
                self.loop
            )
            future.result(timeout=5.0)

        except Exception as e:
            logger_mp.error(f"Error sending state: {e}")

    async def _async_send(self, payload):
        """Async send helper."""
        if self.websocket:
            await self.websocket.send(json.dumps(payload))

    def receive_chunk(self):
        """
        Receive action chunk from server.
        Blocking call.
        Returns: np.ndarray of shape (t, 44) or None.
        """
        if not self.websocket or not self.loop:
            # Try to reconnect if connection is lost
            if not self.websocket:
                logger_mp.info("Attempting to reconnect...")
                if self.connect():
                    logger_mp.info("Reconnected successfully")
                else:
                    return None
            else:
                return None

        try:
            # Receive asynchronously
            future = asyncio.run_coroutine_threadsafe(
                self._async_receive(),
                self.loop
            )
            response = future.result(timeout=30.0)

            if response is None:
                return None

            # Parse response
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response

            # Check if request was successful
            if not data.get("success", True):
                error_msg = data.get("error", "Unknown error")
                logger_mp.error(f"Server returned error: {error_msg}")
                return None

            response_data = data.get("data", data)
            robot_action = None

            if "wh0_robot_action" in response_data:
                robot_action = response_data["wh0_robot_action"]
                logger_mp.debug("Found wh0_robot_action in response_data")
            elif "action_chunk" in response_data:
                robot_action = response_data["action_chunk"]
                logger_mp.debug("Found legacy action_chunk in response_data")
            elif isinstance(response_data, list) and len(response_data) > 0:
                robot_action = response_data
                logger_mp.debug("Using response_data as robot_action directly")
            elif "wh0_robot_action" in data:
                robot_action = data["wh0_robot_action"]
                logger_mp.debug("Found wh0_robot_action in data")
            elif "action_chunk" in data:
                robot_action = data["action_chunk"]
                logger_mp.debug("Found legacy action_chunk in data")
            else:
                logger_mp.warning(f"Unexpected response format. Keys: {list(data.keys())}")
                if "data" in data:
                    logger_mp.warning(f"Data keys: {list(data['data'].keys()) if isinstance(data['data'], dict) else type(data['data'])}")
                return None

            # Convert to numpy array
            try:
                action_array = np.array(robot_action)

                # Validate shape
                if len(action_array.shape) != 2:
                    logger_mp.error(f"Invalid action chunk shape: {action_array.shape}, expected 2D array (t, 44)")
                    return None

                if action_array.shape[1] != 44:
                    logger_mp.error(f"Invalid action chunk shape: {action_array.shape}, expected (t, 44)")
                    return None

                logger_mp.debug(f"Final action chunk shape: {action_array.shape}")
                return action_array
            except Exception as e:
                logger_mp.error(f"Failed to convert robot action to numpy array: {e}")
                return None

        except asyncio.TimeoutError:
            logger_mp.error("Receive timeout: Server did not respond within 30 seconds")
            return None
        except Exception as e:
            logger_mp.error(f"Error receiving chunk: {e}")
            import traceback
            logger_mp.debug(traceback.format_exc())
            return None

    async def _async_receive(self):
        """Async receive helper."""
        if self.websocket:
            try:
                response = await self.websocket.recv()
                return response
            except websockets.exceptions.ConnectionClosed:
                logger_mp.warning("WebSocket connection closed by server")
                self.websocket = None
                return None
            except Exception as e:
                logger_mp.error(f"WebSocket receive error: {e}")
                return None
        return None

    def close(self):
        """Close WebSocket connection."""
        if self.websocket and self.loop:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._async_close(),
                    self.loop
                )
                future.result(timeout=2.0)
            except Exception as e:
                logger_mp.error(f"Error closing connection: {e}")

        if self.loop is not None:
            try:
                if self.loop.is_running():
                    self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception as e:
                logger_mp.debug(f"Error stopping event loop: {e}")

    async def _async_close(self):
        """Async close helper."""
        if self.websocket:
            await self.websocket.close()


class MockArmController:
    """Mock Arm Controller for testing network logic without real robot/sim."""
    def __init__(self, simulation_mode=False, motion_mode=False):
        logger_mp.info("[MockArmController] Initialized.")
        self.q = np.zeros(14)
        self.dq = np.zeros(14)
        self.base_ctrl = self

    def ctrl_dual_arm(self, arm_q_target, arm_tauff_target):
        self.q = arm_q_target

    def get_current_dual_arm_q(self):
        return self.q

    def get_current_dual_arm_dq(self):
        return self.dq

    def ctrl_dual_arm_go_home(self):
        pass

    def speed_gradual_max(self):
        pass

from teleop.televuer.src.televuer.tv_wrapper import TeleVuerWrapper

def instruction_toggle_input_thread(stop_event, instruction_state, instruction_lock, base_instruction, instruction_list=None):
    """
    Toggle instruction when user presses Enter:
    - If instruction_list is provided: cycle through the list (one instruction per Enter press)
    - If instruction_list is None: toggle between base_instruction and a home command
    """
    if instruction_list is not None and len(instruction_list) > 0:
        # List mode: press Enter to cycle through instructions.
        current_index = 0
        with instruction_lock:
            instruction_state["current"] = instruction_list[current_index]
        logger_mp.info(f"[InstructionToggleThread] Started with instruction: {instruction_state['current']}")

        while not stop_event.is_set():
            try:
                user_input = input().strip().lower()
            except EOFError:
                break
            except Exception as e:
                logger_mp.warning(f"[InstructionToggleThread] Input error: {e}")
                break

            if user_input == 'q':
                logger_mp.info("[InstructionToggleThread] Received 'q', stopping...")
                stop_event.set()
                break

            current_index = (current_index + 1) % len(instruction_list)
            with instruction_lock:
                instruction_state["current"] = instruction_list[current_index]
            logger_mp.info(f"[InstructionToggleThread] Instruction switched to: {instruction_state['current']}")
    else:
        # Toggle mode: switch between the task instruction and a home command.
        override_instruction = "Left hand: None. Right hand: return home pose."
        while not stop_event.is_set():
            try:
                user_input = input().strip().lower()
            except EOFError:
                break
            except Exception as e:
                logger_mp.warning(f"[InstructionToggleThread] Input error: {e}")
                break

            if user_input == 'q':
                logger_mp.info("[InstructionToggleThread] Received 'q', stopping...")
                stop_event.set()
                break

            with instruction_lock:
                if instruction_state["use_override"]:
                    instruction_state["current"] = base_instruction
                    instruction_state["use_override"] = False
                    logger_mp.info(f"[InstructionToggleThread] Instruction restored: {instruction_state['current']}")
                else:
                    instruction_state["current"] = override_instruction
                    instruction_state["use_override"] = True
                    logger_mp.info(f"[InstructionToggleThread] Instruction switched: {instruction_state['current']}")


def controller_input_thread(loco_wrapper, camera_config, img_server_ip, stop_event):
    """
    Thread for handling controller input to control robot locomotion.
    """
    try:
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=False, # We want controller
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     display_mode='immersive', # Default
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     )

        logger_mp.info("[ControllerThread] TeleVuerWrapper initialized for controller input.")

        while not stop_event.is_set():
            tele_data = tv_wrapper.get_tele_data()

            if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                loco_wrapper.Damp()
                logger_mp.info("[ControllerThread] Emergency Damp triggered.")

            # Keep locomotion commands within xr_teleoperate's recommended speed range.
            vx = -tele_data.left_ctrl_thumbstickValue[1] * 0.3
            vy = -tele_data.left_ctrl_thumbstickValue[0] * 0.3
            vyaw = -tele_data.right_ctrl_thumbstickValue[0] * 0.3

            if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vyaw) > 0.01:
                loco_wrapper.Move(vx, vy, vyaw)

            time.sleep(0.033) # 30Hz

    except Exception as e:
        logger_mp.error(f"[ControllerThread] Error: {e}")

def run_remote_control(
    server_ip,
    server_port,
    arm_type,
    instruction,
    simulation_mode=False,
    motion_mode=False,
    mock_robot=False,
    img_server_ip='192.168.123.164',
    input_mode='hand',
    instruction_list=None,
    last_step_hand=True,
    debug_dir=None,
):
    if simulation_mode:
        ChannelFactoryInitialize(1)
    else:
        ChannelFactoryInitialize(0)

    loco_wrapper = None

    if not motion_mode and not simulation_mode and not mock_robot:
        try:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")
        except Exception as e:
            logger_mp.warning(f"Failed to enter debug mode: {e}")
    elif motion_mode and not simulation_mode and not mock_robot:
        logger_mp.info("Motion mode enabled. Skipping debug mode entry.")
        try:
            if input_mode == "controller":
                logger_mp.info("Input mode is controller. Initializing LocoClientWrapper...")
                loco_wrapper = LocoClientWrapper()
        except Exception as e:
             logger_mp.error(f"Error during motion initialization: {e}")
             raise e

    logger_mp.info(f"Initializing {arm_type} arm controller (motion_mode={motion_mode})...")

    if mock_robot:
        arm_ctrl = MockArmController(simulation_mode, motion_mode)
        arm_ik = G1_29_ArmIK()
        lower_body_locked_q = G1_29_LOWER_BODY_LOCKED_Q
    elif arm_type == "G1_29":
        arm_ctrl = G1_29_ArmController(simulation_mode=simulation_mode, motion_mode=motion_mode)
        arm_ik = G1_29_ArmIK()
        lower_body_locked_q = G1_29_LOWER_BODY_LOCKED_Q
    elif arm_type == "G1_23":
        arm_ctrl = G1_23_ArmController(simulation_mode=simulation_mode, motion_mode=motion_mode)
        arm_ik = G1_23_ArmIK()
        lower_body_locked_q = G1_23_LOWER_BODY_LOCKED_Q
    elif arm_type == "H1_2":
        arm_ctrl = H1_2_ArmController(simulation_mode=simulation_mode, motion_mode=motion_mode)
        arm_ik = H1_2_ArmIK()
        lower_body_locked_q = H1_2_LOWER_BODY_LOCKED_Q
    elif arm_type == "H1":
        arm_ctrl = H1_ArmController(simulation_mode=simulation_mode, motion_mode=motion_mode)
        arm_ik = H1_ArmIK()
        lower_body_locked_q = H1_LOWER_BODY_LOCKED_Q
    else:
        raise ValueError(f"Unknown arm type: {arm_type}")

    if motion_mode or mock_robot:
        final_arm_ctrl = arm_ctrl
        if not mock_robot:
            final_arm_ctrl.speed_gradual_max()
    else:
        final_arm_ctrl = LockedArmController(arm_ctrl, lower_body_locked_q, simulation_mode)
        final_arm_ctrl.base_ctrl.speed_gradual_max()

    logger_mp.info("Moving arms to initial position...")
    try:
        left_target_pose = np.array([
            [
                    0.6135985255241394,
                    0.014416663907468319,
                    0.7894864082336426,
                    0.06872103943171667
                ],
                [
                    -0.12137202173471451,
                    0.9896730780601501,
                    0.07625927031040192,
                    0.309992399510242
                ],
                [
                    -0.7802342176437378,
                    -0.14261405169963837,
                    0.6090115904808044,
                    -0.20922667410928425
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
        ])

        right_target_pose = np.array([
            [
                    0.7938973903656006,
                    -0.15349611639976501,
                    -0.5883585214614868,
                    0.20543757850187
                ],
                [
                    0.2257864624261856,
                    0.9728481769561768,
                    0.05085831880569458,
                    -0.30886096984495973
                ],
                [
                    0.5645771026611328,
                    -0.1732197403907776,
                    0.8069989681243896,
                    0.1202371058170158
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
        ])

        current_lr_arm_q = final_arm_ctrl.get_current_dual_arm_q()
        current_lr_arm_dq = final_arm_ctrl.get_current_dual_arm_dq()

        arm_joint_solution, arm_feedforward_torque = arm_ik.solve_ik(
            left_target_pose,
            right_target_pose,
            current_lr_arm_q,
            current_lr_arm_dq
        )

        final_arm_ctrl.ctrl_dual_arm(arm_joint_solution, arm_feedforward_torque)
        time.sleep(2.0)
        logger_mp.info("Arms raised to initial position.")
        last_arm_joint_solution = arm_joint_solution

    except Exception as e:
        logger_mp.warning(f"Failed to move to initial position: {e}")

    hand_ctrl = InspireHandDirectController(simulation_mode=simulation_mode or mock_robot)

    img_client = ImageClient()
    logger_mp.info("Image Client initialized.")

    client = VLAClient(server_ip=server_ip, server_port=server_port)
    if not client.connect():
        return

    stop_event = threading.Event()

    instruction_lock = threading.Lock()
    instruction_state = {
        "current": instruction,
        "use_override": False
    }
    instruction_thread = threading.Thread(
        target=instruction_toggle_input_thread,
        args=(stop_event, instruction_state, instruction_lock, instruction, instruction_list),
        daemon=True
    )
    instruction_thread.start()

    if instruction_list is not None and len(instruction_list) > 0:
        logger_mp.info(f"Instruction list mode enabled: {len(instruction_list)} instructions loaded. Press Enter to cycle through instructions, press 'q' to quit.")
    else:
        logger_mp.info("Instruction toggle enabled: press Enter to switch instruction; press Enter again to restore.")

    controller_thread = None
    if motion_mode and loco_wrapper is not None:
        try:
            camera_config = img_client.get_cam_config()
            controller_thread = threading.Thread(target=controller_input_thread,
                                               args=(loco_wrapper, camera_config, img_server_ip, stop_event))
            controller_thread.daemon = True
            controller_thread.start()
            logger_mp.info("Controller input thread started.")
        except Exception as e:
            logger_mp.error(f"Failed to start controller thread: {e}")

    logger_mp.info("Ready to receive VLA chunks...")

    last_left_hand_qpos = [0.998, 1.0, 1.0, 0.999, 0.916, 0.986]
    last_right_hand_qpos = [0.998, 0.998, 0.998, 0.998, 0.99, 0.946]
    left_hand_init = last_left_hand_qpos.copy()
    right_hand_init = last_right_hand_qpos.copy()

    try:
        hand_ctrl.ctrl_dual_hand(left_hand_init, right_hand_init)
        logger_mp.info("Hands initialized to initial position.")
        time.sleep(0.5)
    except Exception as e:
        logger_mp.warning(f"Failed to initialize hand positions: {e}")

    try:
        while True:
            head_img, _ = img_client.get_head_frame()
            if head_img is not None:
                break
            time.sleep(0.1)
        if head_img is None:
            head_img = np.zeros((480, 640, 3), dtype=np.uint8)

        current_arm_q = final_arm_ctrl.get_current_dual_arm_q()
        initial_state = compute_state_in_camera_frame(
            arm_ik, current_arm_q, last_left_hand_qpos, last_right_hand_qpos
        )
        with instruction_lock:
            current_instruction = instruction_state["current"]
        logger_mp.info(f"Sending initial state with instruction: {current_instruction}")
        client.send_state(head_img, current_instruction, initial_state.tolist())

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "head_initial.jpg"), head_img)

        logger_mp.info(f"Instruction: {instruction}")
        logger_mp.info(f"Initial state: {initial_state.tolist()}")

        hand_close_latched = False
        while True:
            robot_action = client.receive_chunk()

            if robot_action is None:
                logger_mp.warning("Received None chunk, exiting...")
                break

            steps_to_execute = min(12, robot_action.shape[0])
            left_final_hand_action = robot_action[steps_to_execute - 1][16:22]
            right_final_hand_action = robot_action[steps_to_execute - 1][38:44]

            for step_idx in range(steps_to_execute):
                step_start = time.time()

                # Wh0 robot action: [Left (22), Right (22)]
                # 0-15: Left Arm (4x4)
                # 16-21: Left Hand (6)
                # 22-37: Right Arm (4x4)
                # 38-43: Right Hand (6)

                action_vec = robot_action[step_idx]

                right_wrist_pose_flat = action_vec[22:38]

                # Only apply hand actions at the LAST step; keep previous state otherwise
                if last_step_hand:
                    if step_idx == steps_to_execute - 1:
                        left_hand_action = left_final_hand_action
                        right_hand_action = right_final_hand_action
                    else:
                        left_hand_action = np.array(last_left_hand_qpos)
                        right_hand_action = np.array(last_right_hand_qpos)
                else:
                    # Apply hand actions at every step
                    left_hand_action = action_vec[16:22]
                    right_hand_action = action_vec[38:44]

                    with instruction_lock:
                        current_instruction = instruction_state["current"]

                    if "return home" in current_instruction.lower() or "home pose" in current_instruction.lower():
                        hand_close_latched = False
                        logger_mp.info("[HandConstraint] Home command detected; resetting hand-close latch.")
                        right_hand_action = np.array(right_hand_init)
                    else:
                        if hand_close_latched:
                            # After closing, keep fingers monotonically closing.
                            right_hand_action = np.minimum(right_hand_action, last_right_hand_qpos)
                            left_hand_action = np.minimum(left_hand_action, last_left_hand_qpos)

                        RIGHT_HAND_THRESHOLD = 1.0
                        if np.all(np.array(right_hand_action) > RIGHT_HAND_THRESHOLD):
                            right_hand_action = np.array(right_hand_init)
                            logger_mp.info(f"Right hand joints all > {RIGHT_HAND_THRESHOLD}, using initial state: {right_hand_init}")

                        RIGHT_HAND_THRESHOLD = 0.7
                        if np.all(np.array(right_hand_action) < RIGHT_HAND_THRESHOLD):
                            logger_mp.info(f"Right hand joints all > {RIGHT_HAND_THRESHOLD}, using initial state: {right_hand_init}")
                            hand_close_latched = True

                right_wrist_pose = right_wrist_pose_flat.reshape(4, 4)

                if step_idx == 0:
                    current_lr_arm_q = locals().get(
                        "last_arm_joint_solution",
                        final_arm_ctrl.get_current_dual_arm_q()
                    )
                else:
                    current_lr_arm_q = last_arm_joint_solution

                current_lr_arm_dq = final_arm_ctrl.get_current_dual_arm_dq()

                arm_joint_solution, arm_feedforward_torque = arm_ik.solve_ik(
                    left_target_pose,
                    right_wrist_pose,
                    current_lr_arm_q,
                    current_lr_arm_dq
                )

                if step_idx == 0 and "last_arm_joint_solution" in locals():
                    chunk_start_blend_alpha = 0.6
                    arm_joint_solution = (
                        (1 - chunk_start_blend_alpha) * last_arm_joint_solution
                        + chunk_start_blend_alpha * arm_joint_solution
                    )

                last_arm_joint_solution = arm_joint_solution

                final_arm_ctrl.ctrl_dual_arm(arm_joint_solution, arm_feedforward_torque)
                hand_ctrl.ctrl_dual_hand(left_hand_action, right_hand_action)

                # Update last executed hand actions (state is the last executed action)
                last_left_hand_qpos = left_hand_action.tolist() if hasattr(left_hand_action, 'tolist') else list(left_hand_action)
                last_right_hand_qpos = right_hand_action.tolist() if hasattr(right_hand_action, 'tolist') else list(right_hand_action)

                elapsed = time.time() - step_start
                time.sleep(0.2)

            while True:
                head_img, _ = img_client.get_head_frame()
                if head_img is not None:
                    break
                time.sleep(0.1)
            if head_img is None:
                # Fallback to black image if no camera
                head_img = np.zeros((480, 640, 3), dtype=np.uint8)
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "head_latest.jpg"), head_img)
            current_arm_q = last_arm_joint_solution
            state_in_camera = compute_state_in_camera_frame(
                arm_ik, current_arm_q, last_left_hand_qpos, last_right_hand_qpos
            )

            with instruction_lock:
                current_instruction = instruction_state["current"]
            client.send_state(head_img, current_instruction, state_in_camera.tolist())

    except KeyboardInterrupt:
        logger_mp.info("Control interrupted by user.")
    except Exception as e:
        logger_mp.error(f"Error during control: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger_mp.info("Returning to home position...")
        try:
            # final_arm_ctrl.ctrl_dual_arm_go_home()
            logger_mp.info("Automatic home motion is disabled; leaving robot state unchanged.")
        except Exception as e:
            logger_mp.error(f"Failed to run shutdown movement: {e}")

        if stop_event:
            stop_event.set()
        if instruction_thread:
            instruction_thread.join(timeout=0.2)
        if controller_thread:
            controller_thread.join(timeout=1.0)

        client.close()
        img_client.close()
        logger_mp.info("Control finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote control G1 robot via VLA Client')
    parser.add_argument('--server-ip', type=str, default='127.0.0.1', help='Server IP address')
    parser.add_argument('--server-port', type=int, default=8765, help='Server port')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'],
                        default='G1_29', help='Select arm controller')
    parser.add_argument('--instruction', type=str, default="Left hand: None. Right hand: Pick the white and black toy.", help='Instruction for the task')
    parser.add_argument('--sim', action='store_true', help='Enable simulation mode')
    parser.add_argument('--motion', action='store_true', help='Enable motion mode (AI sport)')
    parser.add_argument('--mock-robot', action='store_true', help='Enable mock robot controller for network testing')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='Image server IP (Robot IP)')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--instruction-list', type=str, default='instructions.txt', help='Path to instruction list file (one instruction per line). Press Enter to cycle through instructions.')
    parser.add_argument('--last-step-hand', action='store_true',
                        help='Only apply hand actions at the last step of each chunk (default behavior). '
                             'Without this flag, hand actions are executed at every step.')
    parser.add_argument('--debug-dir', type=str, default=None, help='Optional directory for latest camera debug frames')

    args = parser.parse_args()

    # Determine instruction mode: instruction_list takes precedence over single instruction
    instruction_list = None
    effective_instruction = args.instruction  # default fallback

    if args.instruction_list:
        if os.path.exists(args.instruction_list):
            with open(args.instruction_list, 'r', encoding='utf-8') as f:
                instruction_list = [line.strip() for line in f.readlines() if line.strip()]
            logger_mp.info(f"Loaded {len(instruction_list)} instructions from {args.instruction_list}")
            # Use first instruction as the initial one
            if len(instruction_list) > 0:
                effective_instruction = instruction_list[0]
        else:
            logger_mp.warning(f"Instruction list file not found: {args.instruction_list}, falling back to single instruction mode")

    run_remote_control(
        args.server_ip,
        args.server_port,
        args.arm,
        effective_instruction,
        args.sim,
        args.motion,
        args.mock_robot,
        args.img_server_ip,
        args.input_mode,
        instruction_list,
        args.last_step_hand,
        args.debug_dir,
    )
