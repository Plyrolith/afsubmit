from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any


# Dictionary of block flags and their corresponding bit values

BLOCKFLAGS: dict[str, int] = {
    "numeric": 1 << 0,
    "varcapacity": 1 << 1,
    "multihost": 1 << 2,
    "masteronslave": 1 << 3,
    "dependsubtask": 1 << 4,
    "skipthumbnails": 1 << 5,
    "skipexistingfiles": 1 << 6,
    "checkrenderedfiles": 1 << 7,
    "slavelostignore": 1 << 8,
    "appendedtasks": 1 << 9,
}


# Store for Afanasy config variables

VARS: dict[str, str] | None = None


# Utility functions


def checkBlockFlag(i_flags: int, i_name: str) -> bool:
    """
    Check if a block flag is set in given flags value.

    Parameters:
        - i_flags (int): Flag value to check
        - i_name (str): Name of the flag to check

    Returns:
        - bool: True if flag is set, False otherwise
    """
    if i_name not in BLOCKFLAGS:
        print(f"AFERROR: block flag {i_name} does not exist")
        print(f"Existing flags are: {BLOCKFLAGS}")
        return False

    return i_flags & BLOCKFLAGS[i_name]


def checkClass(name: str, folder: str) -> bool:
    """
    Check if a python module exists in given Afanasy root sub folder.

    Parameters:
        - name (str): Name of the module to check
        - folder (str): Folder to check

    Returns:
        - bool: True if module exists, False otherwise
    """
    af_root = getVar("AF_ROOT")
    if not af_root:
        print(f"Cannot check module {name}, AF_ROOT is not set")
        return False

    directory = Path(af_root, "python", folder)
    if Path(directory, name).with_suffix(".py") in directory.iterdir():
        return True

    return False


def checkRegExp(pattern: str) -> bool:
    """
    Check if given string is a valid regular expression pattern.

    Parameters:
        - pattern (str): Regular expression pattern to check

    Returns:
        - bool: True if pattern is valid, False otherwise
    """
    if len(pattern) == 0:
        return False
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def genHeader(i_data_size: int) -> bytearray:
    """
    Generate a header for a message to send to the server.

    Parameters:
        - i_data_size (int): Size of the data to send

    Returns:
        - bytearray: Header
    """
    data = f"AFANASY {i_data_size} JSON"
    return bytearray(data, "utf-8")


def getVar(i_name: str, default: Any = "") -> str | None:
    """
    Get an CGRU config variable. Try to load all config files if not already loaded.
    Return defaults if variable is not found.

    Parameters:
        - i_name (str): Name of the variable to get
        - default (Any): Default value to return if variable is not found

    Returns:
        - str | None: Value of the variable, if found
    """
    if VARS is None:
        loadConfigs()

    return VARS.get(i_name, default)


def loadConfigs() -> dict[str, str]:
    """
    Load CGRU config files and store variables in global VARS dictionary.

    Returns:
        - dict[str, str]: Dictionary of config variables
    """
    global VARS
    config_files: list[Path] = []

    # Add default config file
    platform = sys.platform
    cgru_location = os.getenv("CGRU_LOCATION")
    if cgru_location:
        config_files.append(Path(cgru_location, "config.json"))

    # Try default config locations
    else:
        if platform.startswith("win"):
            config_files.append(Path("C:/cgru/config.json"))
            config_files.append(Path("C:/Program Files/cgru/config.json"))
            config_files.append(Path("C:/Program Files (x86)/cgru/config.json"))
        elif platform.startswith("linux"):
            config_files.append(Path("/opt/cgru/config.json"))
        elif platform.startswith("darwin"):
            config_files.append(Path("/Applications/cgru/config.json"))

    # Add custom config file
    custom_location = os.getenv("CGRU_CUSTOM_CONFIG")
    if custom_location:
        config_files.append(Path(custom_location))

    # Add home config file
    user_home = os.getenv("APPDATA", os.getenv("HOME"))
    if user_home:
        if platform.startswith("win"):
            home_config = "cgru"
        else:
            home_config = ".cgru"
        config_files.append(Path(user_home, home_config, "config.json"))

    # Load config files
    vars = {}
    for config_file in config_files:
        if config_file.is_file():
            print(f"Loading config file: {config_file}")
            with open(config_file, "r") as file:
                config: dict = json.load(file)
                vars |= config.get("cgru_config", {})

    # Remove empty keys
    vars.pop("", None)

    VARS = vars
    return vars


def mapServer(i_path: Path | str) -> str:
    """
    Map given path to server if path mapping is available from the CGRU config.

    Parameters:
        - i_path (Path | str): Path to map

    Returns:
        - str: Mapped path
    """
    pathsmap = getVar("pathsmap")
    if pathsmap:
        i_path = Path(str(i_path).replace("\\", "/"))
        for paths in pathsmap:
            if len(paths) != 2:
                print("ERROR: Pathmap is not a pair")
                continue

            client_path = Path(paths[0].replace("\\", "/"))
            server_path = Path(paths[1].replace("\\", "/"))

            # Remap path
            if i_path.is_relative_to(client_path):
                print(f"Remapping path: {i_path}")
                i_path = i_path.relative_to(client_path)
                i_path = Path(server_path, i_path)
                print(f"Remapped path: {i_path}")
                break

        else:
            print("No path mapping found")

    return str(i_path)


def setBlockFlag(i_flags: int, i_name: str) -> int:
    """
    Takes a block's flag value and modifies it by setting the given flag.

    Parameters:
        - i_flags (int): Flag value to modify
        - i_name (str): Name of the flag to set

    Returns:
        - int: Resulting modified flag value
    """
    if i_name not in BLOCKFLAGS:
        print(f"AFERROR: block flag {i_name} does not exist")
        print(f"Existing flags are: {BLOCKFLAGS}")
        return i_flags

    elif i_name == "appendedtasks":
        print(f"AFERROR: block flag {i_name} is read-only")
        return i_flags

    return i_flags | BLOCKFLAGS[i_name]


def sendServer(
    i_data: str,
    i_host: str = "afanasy",
    i_port: int = 51000,
    i_verbose: bool = False,
    i_without_answer: bool = False,
) -> tuple[bool, object | None]:
    """
    Send data to the server.

    Parameters:
        - i_data (str): Data to send
        - i_host (str): Host to connect to
        - i_port (int): Port to connect to
        - i_verbose (bool): Print verbose information
        - i_without_answer (bool): Do not wait for an answer

    Returns:
        - tuple[bool, object | None]: Success status and answer
    """
    size = len(i_data)
    header = genHeader(size)
    i_data = header + bytearray(i_data, "utf-8")
    datalen = len(i_data)

    af_socket = None
    error_msg = ""
    address_infos = []

    try:
        address_infos = socket.getaddrinfo(
            host=i_host,
            port=i_port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except:  # TODO: Too broad exception clause
        print(f"Can`t solve {i_host}: {sys.exc_info()[1]}")

    for address_info in address_infos:
        family, type, proto, _, address = address_info
        if i_verbose:
            print(f"Trying to connect to {address[0]}")
        try:
            af_socket = socket.socket(family, type, proto)
        except:  # TODO: Too broad exception clause
            if error_msg != "":
                error_msg += "\n"
            error_msg += f"{address[0]}: {sys.exc_info()[1]}"
            af_socket = None
            continue
        try:
            af_socket.connect(address)
        except:  # TODO: Too broad exception clause
            if error_msg != "":
                error_msg += "\n"
            error_msg += f"{address[0]}: {sys.exc_info()[1]}"
            af_socket.close()
            af_socket = None
            continue
        break

    if af_socket is None:
        print("Could not open socket")
        print(error_msg)
        return False, None

    if i_verbose:
        print(f"afnetwork.sendServer: send {datalen} bytes")

    # s.sendall( i_data) #<<< !!! May not work !!!!

    total_send = 0
    while total_send < len(i_data):
        sent = af_socket.send(i_data[total_send:])
        if sent == 0:
            af_socket.close()
            print("Error: Unable send data to socket")
            return False, None
        total_send += sent

    data = b""
    msg_len = None
    while True:
        data_buffer = af_socket.recv(4096)

        if not data_buffer:
            break

        data += data_buffer

        if msg_len is None:
            dataStr = toStr(data)
            if dataStr.find("AFANASY") != -1 and dataStr.find("JSON") != -1:
                msg_len = dataStr[: dataStr.find("JSON") + 4]
                msg_len = len(msg_len) + int(msg_len.split(" ")[1])

        if i_verbose:
            print(f"Received {len(data)} of {msg_len} bytes")

        if msg_len is not None:
            if len(data) >= msg_len:
                break

    af_socket.close()

    struct = None

    if i_without_answer is True:
        return True, struct

    try:
        if not isinstance(data, str):
            data = toStr(data)
        if msg_len is not None:
            data = data[data.find("JSON") + 4 : msg_len]
        struct = json.loads(data, strict=False)
    except:  # TODO: Too broad exception clause
        print("afnetwork.py: Received data:")
        print(data)
        print("JSON loads error:")
        print(str(sys.exc_info()[1]))
        struct = None

    return True, struct


def toStr(data: str | bytes) -> str:
    """
    Convert given data to a UTF-8 string, if necessary.

    Parameters:
        - data (str | bytes): Data to convert

    Returns:
        - str: UTF-8 string
    """
    if isinstance(data, str):
        return data

    return str(data, "utf-8", "replace")


# Job sub classes & main class (hierarchical structure)


class Task:
    """Task class, lowest level of the hierarchy"""

    name: str = ""
    command: str | None = None
    files: list[str] | None = None
    environment: dict[str, str] | None = None

    def __init__(self, taskname: str | None = None):
        self.setName(taskname)

    def asDict(self) -> dict:
        """
        Return this task as a dictionary, using only fields with filled values.

        Returns:
            - dict: Dictionary representation of this task
        """
        data_dict = {}
        for key in self.__annotations__.keys():
            value = getattr(self, key)
            if value is not None:
                data_dict[key] = value

        return data_dict

    def setCommand(self, command: str, TransferToServer: bool = True):
        """
        Set the command of this task.

        Parameters:
            - command (str): Command to set
            - TransferToServer (bool): Whether to map the command's paths to the server
        """
        if TransferToServer:
            command = mapServer(command)

        self.command = command

    def setEnv(self, i_name: str, i_value: str, i_transfer_to_server: bool = True):
        """
        Set an environment variable for this task.

        Parameters:
            - i_name (str): Name of the environment variable
            - i_value (str): Value of the environment variable
            - i_transfer_to_server (bool): Whether to map the environment variable's
              paths to the server
        """
        if self.environment is None:
            self.environment = {}

        if i_transfer_to_server:
            i_value = mapServer(i_value)

        self.environment[i_name] = i_value

    def setFiles(self, files: list[str], TransferToServer: bool = True):
        """
        Add files to this task.

        Parameters:
            - files (list[str]): List of files to add
            - TransferToServer (bool): Whether to map the files' paths to the server
        """
        if self.files is None:
            self.files = []

        for file in files:
            if TransferToServer:
                file = mapServer(file)

            self.files.append(file)

    def setName(self, name: str):
        """
        Set the name of this task.

        Parameters:
            - name (str): Name of the task
        """
        if name:
            self.name = name


class Block:
    """Block class, sub-level of the job hierarchy"""

    capacity: int | None = None
    capacity_coeff_max: int | None = None
    capacity_coeff_min: int | None = None
    command: str | None = None
    command_post: str | None = None
    command_pre: str | None = None
    depend_mask: str | None = None
    environment: dict[str, str] | None = None
    errors_avoid_host: int | None = None
    errors_retries: int | None = None
    errors_forgive_time: int | None = None
    errors_task_same_host: int | None = None
    file_size_max: int | None = None
    file_size_min: int | None = None
    files: list[str] | None = None
    flags: int | None = None
    frame_first: int | None = None
    frame_last: int | None = None
    frames_inc: int | None = None
    frames_per_task: int | None = None
    hosts_mask: str | None = None
    hosts_mask_exclude: str | None = None
    max_running_tasks: int | None = None
    max_running_tasks_per_host: int | None = None
    multihost_max: int | None = None
    multihost_max_wait: int | None = None
    multihost_min: int | None = None
    multihost_service: str | None = None
    multihost_service_wait: int | None = None
    name: str = "block"
    need_cpu_cores: int | None = None
    need_cpu_freq_cores: int | None = None
    need_cpu_freq_mgz: int | None = None
    need_gpu_mem_mb: int | None = None
    need_hdd: int | None = None
    need_memory: int | None = None
    need_power: int | None = None
    need_properties: str | None = None
    parser_coeff: float | None = None
    sequential: int | None = None
    service: str = "generic"
    task_max_run_time: int | None = None
    task_min_run_time: int | None = None
    task_progress_change_timeout: int | None = None
    tasks: list[Task] | None = None
    tasks_depend_mask: str | None = None
    tasks_name: str | None = None
    tickets: dict[str, int] | None = None
    parser: str | None = None
    working_directory: str | None = None

    def __init__(self, blockname: str | None = None, service: str | None = None):
        self.flags = 0
        self.tasks = []
        if blockname:
            self.setName(blockname)
        self.capacity = getVar("af_task_default_capacity", 1000)

        working_directory = os.getenv("PWD", os.getcwd())
        if working_directory:
            self.working_directory = working_directory

        self.service = getVar("af_task_default_service", "generic")
        if service:
            self.service = service
            if self.setService(service):
                __import__("services", globals(), locals(), [self.service])
                parser = eval((f"services.{self.service}.parser"))
                self.setParser(parser)

    def addTicket(self, i_name: str, i_count: int):
        """
        Add a ticket to this block.

        Parameters:
            - i_name (str): Name of the ticket
            - i_count (int): Count of the ticket
        """
        if self.tickets is None:
            self.tickets = {}

        self.tickets[i_name] = i_count

    def asDict(self) -> dict:
        """
        Return this block as a dictionary, using only fields with filled values.

        Returns:
            - dict: Dictionary representation of this block
        """
        data_dict = {}
        for key in self.__annotations__.keys():
            if key == "tasks":
                value = [task.asDict() for task in self.tasks]
            else:
                value = getattr(self, key)
            if value is not None:
                data_dict[key] = value

        return data_dict

    def checkRenderedFiles(self, i_size_min: int = -1, i_size_max: int = -1):
        """
        Set the check rendered files flag, as well as required file size properties for
        this block.

        Parameters:
            - i_size_min (int): Minimum file size in MB
            - i_size_max (int): Maximum file size in MB
        """
        self.flags = setBlockFlag(self.flags, "checkrenderedfiles")
        if i_size_min > 0:
            self.file_size_min = i_size_min
        if i_size_max > 0:
            self.file_size_max = i_size_max

    def fillTasks(self):
        """
        Deprecated.
        """
        pass

    def setCapacity(self, capacity: int):
        """
        Set the capacity of this block.

        Parameters:
            - capacity (int): Capacity
        """
        if capacity > 0:
            self.capacity = capacity

    def setCmdPre(self, command_pre: str, TransferToServer: bool = True):
        """
        Set the pre-command of this block.

        Parameters:
            - command_pre (str): Pre-command
            - TransferToServer (bool): Whether to map included paths to the server
        """
        if TransferToServer:
            command_pre = mapServer(command_pre)

        self.command_pre = command_pre

    def setCmdPost(self, command_post: str, TransferToServer: bool = True):
        """
        Set the post-command of this block.

        Parameters:
            - command_post (str): Post-command
            - TransferToServer (bool): Whether to map included paths to the server
        """
        if TransferToServer:
            command_post = mapServer(command_post)

        self.command_post = command_post

    def setCommand(
        self,
        command: str,
        prefix: bool = True,
        TransferToServer: bool = True,
    ):
        """
        Set the command of this block.

        Parameters:
            - command (str): Command
            - prefix (bool): Whether to add the command prefix
            - TransferToServer (bool): Whether to map included paths to the server
        """
        if prefix:
            command = os.getenv("AF_CMD_PREFIX", getVar("af_cmdprefix")) + command
        if TransferToServer:
            command = mapServer(command)

        self.command = command

    def setDependMask(self, value: str):
        """
        Set the dependency mask (regex pattern) for this block.
        This block will be executed if all blocks matching this pattern are finished.

        Parameters:
            - value (str): Dependency mask pattern
        """
        if checkRegExp(value):
            self.depend_mask = value

    def setDependSubTask(self):
        """
        Set the sub task dependency flag for this block.
        """
        self.flags = setBlockFlag(self.flags, "dependsubtask")

    def setEnv(self, i_name: str, i_value: str, i_transfer_to_server: bool = True):
        """
        Set an environment variable for this block.

        Parameters:
            - i_name (str): Name of the environment variable
            - i_value (str): Value of the environment variable
            - i_transfer_to_server (bool): Whether to map value paths to the server
        """
        if self.environment is None:
            self.environment = {}

        if i_transfer_to_server:
            i_value = mapServer(i_value)

        self.environment[i_name] = i_value

    def setErrorsAvoidHost(self, value: int):
        """
        Set the number of errors that lead to avoiding a host for this block.

        Parameters:
            - value (int): Number of errors
        """
        self.errors_avoid_host = value

    def setErrorsForgiveTime(self, value: int):
        """
        Set the time to forgive errors for this block.

        Parameters:
            - value (int): Time to forgive errors in seconds
        """
        self.errors_forgive_time = value

    def setErrorsRetries(self, value: int):
        """
        Set the number of retries after an error for this block.

        Parameters:
            - value (int): Number of retries
        """
        self.errors_retries = value

    def setErrorsTaskSameHost(self, value: int):
        """
        Set the number of task errors that lead to avoiding a host for this block.

        Parameters:
            - value (int): Number of task errors
        """
        self.errors_task_same_host = value

    def setFiles(self, files: list[str], TransferToServer: bool = True):
        """
        Add files of this block.

        Parameters:
            - files (list[str]): Files to add
            - TransferToServer (bool): Whether to map file paths to the server
        """
        if self.files is None:
            self.files = []

        for file in files:
            if TransferToServer:
                file = mapServer(file)

            self.files.append(file)

    def setFramesPerTask(self, value: int):
        """
        Set the frames per task of this block.

        Parameters:
            - value (int): Frames per task
        """
        self.frames_per_task = int(value)

    def setHostsMask(self, value: str):
        """
        Set the hosts mask (regex pattern) for this block.
        Only hosts matching this pattern will be used.

        Parameters:
            - value (str): Hosts mask pattern
        """
        if checkRegExp(value):
            self.hosts_mask = value

    def setHostsMaskExclude(self, value: str):
        """
        Set the hosts mask exclude (regex pattern) for this block.
        Hosts matching this pattern will not be used.

        Parameters:
            - value (str): Hosts mask exclude pattern
        """
        if checkRegExp(value):
            self.hosts_mask_exclude = value

    def setMaxRunTasksPerHost(self, value: int):
        """
        Set the maximum number of simultaneously running tasks per host for this block.

        Parameters:
            - value (int): Maximum number of running tasks per host
        """
        if value >= 0:
            self.max_running_tasks_per_host = int(value)

    def setMaxRunningTasks(self, value: int):
        """
        Set the maximum number of simultaneously running tasks for this block.

        Parameters:
            - value (int): Maximum number of running tasks
        """
        if value >= 0:
            self.max_running_tasks = int(value)

    def setMultiHost(
        self,
        h_min: int,
        h_max: int,
        h_max_wait: int,
        master_on_slave: bool = False,
        service: str | None = None,
        service_wait: int = -1,
    ):
        """
        Set up multihost for this block.

        Parameters:
            - h_min (int): Minimum number of hosts
            - h_max (int): Maximum number of hosts
            - h_max_wait (int): Maximum wait time for hosts
            - master_on_slave (bool): Enable master on slave
            - service (str): Service name
            - service_wait (int): Service wait time
        """
        if h_min < 1:
            print("Error: Block::setMultiHost: Minimum must be greater than zero")
            return False

        if h_max < h_min:
            print("Block::setMultiHost: Maximum must be greater or equal then minimum")
            return False

        if master_on_slave and service is None:
            print(
                "Error: Block::setMultiHost: Master in slave is enabled but "
                "service was not specified"
            )
            return False

        self.flags = setBlockFlag(self.flags, "multihost")
        self.multihost_min = h_min
        self.multihost_max = h_max
        self.multihost_max_wait = h_max_wait

        if master_on_slave:
            self.flags = setBlockFlag(self.flags, "masteronslave")

        if service:
            self.multihost_service = service

        if service_wait > 0:
            self.multihost_service_wait = service_wait

    def setName(self, value: str):
        """
        Set this block's name.
        Each Block has an unique name. If new Block added to Job which the name already
        exists, Job change it’s name by adding a number. Blocks dependence bases on
        their names and depend masks to match it.

        Parameters:
            - value (str): Name
        """
        if value:
            self.name = value

    def setNeedCPUCores(self, value: int):
        """
        Set the required amount of CPU cores for this block.

        Parameters:
            - value (int): Required amount of CPU cores
        """
        if value > 0:
            self.need_cpu_cores = int(value)

    def setNeedCPUFreqCores(self, value: int | float):
        """
        Set the required CPU frequency per core in GHz for this block.

        Parameters:
            - value (int | float): Required CPU frequency per core in GHz
        """
        if value > 0:
            self.need_cpu_freq_cores = int(value * 1000.0)

    def setNeedCPUFreqGHz(self, value: int | float):
        """
        Set the required CPU frequency in GHz for this block.

        Parameters:
            - value (int | float): Required CPU frequency in GHz
        """
        if value > 0:
            self.need_cpu_freq_mgz = int(value * 1000.0)

    def setNeedGPUMemGB(self, value: int | float):
        """
        Set the required GPU memory in GB for this block.

        Parameters:
            - value (int | float): Required GPU memory in GB
        """
        if value > 0:
            self.need_gpu_mem_mb = int(value * 1024.0)

    def setNeedMemory(self, value: int):
        """
        Set the required memory in MB for this block.

        Parameters:
            - value (int): Required memory in MB
        """
        if value > 0:
            self.need_memory = int(value)

    def setNeedPower(self, value: int):
        """
        Set the required power for this block.

        Parameters:
            - value (int): Required power
        """
        if value > 0:
            self.need_power = int(value)

    def setNeedProperties(self, value: str):
        """
        Set required properties (regex pattern) for this block.

        Parameters:
            - value (str): Required properties pattern
        """
        if checkRegExp(value):
            self.need_properties = value

    def setNeedHDD(self, value: int):
        """
        Set the required disk space in MB for this block.

        Parameters:
            - value (int): Required disk space in MB
        """
        if value > 0:
            self.need_hdd = int(value)

    def setNumeric(
        self,
        start: int = 1,
        end: int = 1,
        pertask: int = 1,
        increment: int = 1,
    ):
        """
        Set the numeric frame parameters of this block.

        Parameters:
            - start (int): First frame
            - end (int): Last frame
            - pertask (int): Frames per task
            - increment (int): Frame increment
        """
        if len(self.tasks):
            print("Error: Block.setNumeric: Block already has tasks")
            return
        if end < start:
            print(f"Error: Block.setNumeric: end < start ({end} < {start})")
            end = start
        if pertask < 1:
            print(f"Error: Block.setNumeric: pertask < 1 ({pertask} < 1)")
            pertask = 1

        self.flags = setBlockFlag(self.flags, "numeric")
        self.frame_first = start
        self.frame_last = end
        self.frames_per_task = pertask
        self.frames_inc = increment

    def setParser(self, parser, nocheck: bool = False):
        """
        Set the parser of this block.

        Parameters:
            - parser (str): Parser to set
            - nocheck (bool): Whether to check if the parser exists
        """
        if parser:
            if not nocheck:
                if not checkClass(parser, "parsers"):
                    if parser != "none":
                        print(f'Error: Unknown parser "{parser}", setting to "none"')
                        parser = "none"
            self.parser = parser

    def setParserCoeff(self, value: float):
        """
        Set parser coefficient for this block.

        Parameters:
            - value (float): Parser coefficient
        """
        self.parser_coeff = value

    def setSequential(self, value: int):
        """
        Set the sequential value of this block.
        By default, sequential is 1, tasks will be solved from the first to the last one
        by one. If this parameter is -1, tasks will be solved from the last to the first
        one by one. If this parameter is greater than 1 or less than -1, 10 for example,
        tasks with every 10 frame will be solved at first, than other tasks. If -10,
        every 10 frame but from the end. Important thing that task frame is used for
        sequential calculation, not task number. If sequential is 0, always middle task
        will be solved. For example if frame range is 1-100, tasks solving order will
        be: 1,100,50,25,75 and so on.

        Parameters:
            - value (int): Sequential value
        """
        self.sequential = int(value)

    def setService(self, service: str, nocheck: bool = False) -> bool:
        """
        Set the service of this block.

        Parameters:
            - service (str): Service to set
            - nocheck (bool): Whether to check if the service exists

        Returns:
            - bool: Whether the service was set successfully
        """
        if service:
            result = True
            if not nocheck:
                if not checkClass(service, "services"):
                    print(f'Error: Unknown service "{service}", setting to "generic"')
                    service = "generic"
                    result = False

            self.service = service
            return result

        return False

    def setSlaveLostIgnore(self):
        """
        Set the flag for ignoring lost clients for this block, if multihost is enabled.
        """
        if not checkBlockFlag(self.flags, "multihost"):
            print("Block::setSlaveLostIgnore: Block is not multihost")
            return
        self.flags = setBlockFlag(self.flags, "slavelostignore")

    def setTaskMaxRunTime(self, value: int):
        """
        Set the maximum task run time in seconds for this block.

        Parameters:
            - value (int): Maximum task run time in seconds
        """
        if value > 0:
            self.task_max_run_time = int(value)

    def setTaskMinRunTime(self, value: int):
        """
        Set the minimum task run time in seconds for this block.

        Parameters:
            - value (int): Minimum task run time in seconds
        """
        if value > 0:
            self.task_min_run_time = int(value)

    def setTaskProgressChangeTimeout(self, value: int):
        """
        Set the task progress change timeout in seconds for this block.

        Parameters:
            - value (int): Task progress change timeout in seconds
        """
        if value > 0:
            self.task_progress_change_timeout = int(value)

    def setTasksDependMask(self, value: str):
        """
        Set the task dependency mask (regex pattern) for this block.
        This block will be executed if all tasks matching this pattern are finished.

        Parameters:
            - value (str): Task dependency mask pattern
        """
        if checkRegExp(value):
            self.tasks_depend_mask = value

    def setTasksMaxRunTime(self, value: int):
        """
        Set the maximum task run time in seconds for this block.
        (DEPRECATED, use setTaskMaxRunTime instead.)

        Parameters:
            - value (int): Maximum task run time in seconds
        """
        self.setTaskMaxRunTime(value)

    def setTasksName(self, value: str):
        """
        Set this block's tasks name.

        Parameters:
            - value (str): Tasks name
        """
        self.tasks_name = value

    def setVariableCapacity(self, capacity_coeff_min: int, capacity_coeff_max: int):
        """
        Set the variable capacity of this block.

        Parameters:
            - capacity_coeff_min (int): Minimum capacity coefficient
            - capacity_coeff_max (int): Maximum capacity coefficient
        """
        if capacity_coeff_min >= 0 or capacity_coeff_max >= 0:
            self.capacity_coeff_min = int(capacity_coeff_min)
            self.capacity_coeff_max = int(capacity_coeff_max)

    def setWorkingDirectory(
        self,
        working_directory: str,
        TransferToServer: bool = True,
    ):
        """
        Set the working directory of this block.

        Parameters:
            - working_directory (str): Working directory
            - TransferToServer (bool): Whether to map the path to the server
        """
        if TransferToServer:
            working_directory = mapServer(working_directory)

        self.working_directory = working_directory

    def skipExistingFiles(self, i_size_min: int = -1, i_size_max: int = -1):
        """
        Set the skip existing files flag, as well as required file size properties for
        this block.

        Parameters:
            - i_size_min (int): Minimum file size in MB
            - i_size_max (int): Maximum file size in MB
        """
        self.flags = setBlockFlag(self.flags, "skipexistingfiles")
        if i_size_min > 0:
            self.file_size_min = i_size_min
        if i_size_max > 0:
            self.file_size_max = i_size_max

    def skipThumbnails(self):
        """
        Set the skip thumbnails flag for this block. Thumbnails will not be generated.
        """
        self.flags = setBlockFlag(self.flags, "skipthumbnails")


class Job:
    """Job class, highest level of abstraction in Afanasy"""

    _host: str = "afanasy"
    _port: int = 51000
    annotation: str | None = None
    blocks: list[Block] | None = None
    branch: str | None = None
    command_post: str | None = None
    command_pre: str | None = None
    department: str | None = None
    depend_mask: str | None = None
    depend_mask_global: str | None = None
    description: str | None = None
    folders: dict[str, str] | None = None
    host_name: str | None = None
    hosts_mask: str | None = None
    hosts_mask_exclude: str | None = None
    ignorenimby: bool | None = None
    ignorepaused: bool | None = None
    maintenance: bool | None = None
    max_running_tasks: int | None = None
    max_running_tasks_per_host: int | None = None
    name: str = "noname"
    need_os: str | None = None
    need_properties: str | None = None
    offline: bool | None = None
    pools: list[str] | None = None
    ppa: bool | None = None
    priority: int = 99
    project: str | None = None
    time_creation: int | None = None
    time_life: int | None = None
    time_wait: int | None = None
    try_this_blocks_num: list[int] | None = None
    try_this_tasks_num: list[int] | None = None
    user_name: str | None = None

    def __init__(
        self,
        jobname: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ):
        self.setName(jobname)
        self.blocks = []

        if host:
            self._host = host
        else:
            self._host = getVar("af_servername", "afanasy")

        if port:
            self._port = port
        else:
            self._port = int(getVar("af_serverport", 51000))

        self.setUserName(getVar("USERNAME", os.getlogin()))
        self.host_name = getVar("HOSTNAME", socket.gethostname())

        priority = getVar("af_priority", None)
        if priority is not None:
            self.setPriority(priority)

        self.time_creation = int(time.time())

    def asDict(self) -> dict:
        """
        Return this job as a dictionary, using only public fields with filled values.

        Returns:
            - dict: Dictionary representation of this job
        """
        data_dict = {}
        for key in self.__annotations__.keys():
            if key.startswith("_"):
                continue
            elif key == "blocks":
                value = [block.asDict() for block in self.blocks]
            else:
                value = getattr(self, key)
            if value is not None:
                data_dict[key] = value

        return data_dict

    def checkJob(self) -> bool:
        """
        Check if this job is valid by checking if all unflagged blocks have tasks.

        Returns:
            - bool: Whether this job is valid
        """
        for block in self.blocks:
            if block.flags == 0 and not block.tasks:
                return False

        return True

    def fillBlocks(self):
        """
        Deprecated.
        """
        return

    def pause(self):
        """
        Submit this job in paused state.
        (DEPRECATED: use setPaused() instead.)
        """
        self.setPaused()

    def offLine(self):
        """
        Submit this job in offline state.
        (DEPRECATED: use setOffline() instead.)
        """
        self.setOffline()

    def output(self):
        """
        Generate a serialized JSON representation of this job and print it to stdout.
        """
        print(json.dumps(self.asDict(), sort_keys=True, indent=4))

    def send(self, verbose=False) -> tuple[bool, object | None]:
        """
        Send this job to the server.

        Parameters:
            - verbose (bool): Whether to print the response from the server

        Returns:
            - tuple[bool, object | None]: Success status and answer from the server
        """
        if not self.blocks:
            print("Error: Job has no blocks")

        if not self.checkJob():
            return False

        # Set folder if empty
        if self.folders is None:
            self.folders = {}

            # Try to set output folder from files
            for block in self.blocks:
                if block.files:
                    parent_name = Path(block.files[0]).parent.name
                    output_folder = Path(block.working_directory, parent_name)
                    self.folders[block.name] = str(output_folder)

        # Set branch if empty
        if self.branch is None:
            if "output" in self.folders:
                self.branch = self.folders["output"]
            elif "input" in self.folders:
                self.branch = self.folders["input"]
            else:
                self.branch = self.blocks[0].working_directory

        data_dict = {"job": self.asDict()}

        return sendServer(json.dumps(data_dict), self._host, self._port, verbose)

    def setAnnotation(self, value: str):
        """
        Set the annotation for this job.

        Parameters:
            - value (str): Annotation
        """
        self.annotation = value

    def setAnyOS(self):
        """
        Set this job to be able to run on any OS.
        """
        self.need_os = ""

    def setBranch(self, i_branch: str, i_transferToServer: bool = True):
        """
        Set the branch for this job.

        Parameters:
            - i_branch (str): Branch path
            - i_transferToServer (bool): Whether to map the path to the server
        """
        if i_transferToServer:
            i_branch = mapServer(i_branch)

        self.branch = i_branch

    def setCmdPre(self, command: str, TransferToServer: bool = True):
        """
        Set the command to be executed before the job starts.

        Parameters:
            - command (str): Command to be executed
            - TransferToServer (bool): Whether to map paths in the command to the server
        """
        if TransferToServer:
            command = mapServer(command)

        self.command_pre = command

    def setCmdPost(self, command: str, TransferToServer: bool = True):
        """
        Set the command to be executed after the job finishes.

        Parameters:
            - command (str): Command to be executed
            - TransferToServer (bool): Whether to map paths in the command to the server
        """
        if TransferToServer:
            command = mapServer(command)

        self.command_post = command

    def setDepartment(self, department: str):
        """
        Set the name of the department which submitted this job.

        Parameters:
            - department (str): Name of the department (compositing, modeling, etc.)
        """
        if department:
            self.department = department

    def setDependMask(self, value: str):
        """
        Set the dependency mask (regex pattern) for this job.
        This job will not start until all jobs that match this pattern are completed.

        Parameters:
            - value (str): Dependency mask
        """
        if checkRegExp(value):
            self.depend_mask = value

    def setDependMaskGlobal(self, value: str):
        """
        Set the global dependency mask (regex pattern) for this job.
        This job will not start until all jobs that match this pattern are completed.

        Parameters:
            - value (str): Global dependency mask
        """
        if checkRegExp(value):
            self.depend_mask_global = value

    def setDescription(self, value: str):
        """
        Set the description for this job.

        Parameters:
            - value (str): Description
        """
        self.description = value

    def setFolder(self, i_name: str, i_folder: str, i_transferToServer: bool = True):
        """
        Add a folder to this job.

        Parameters:
            - i_name (str): Name of the folder
            - i_folder (str): Path to the folder
            - i_transferToServer (bool): Whether to map the folder path to the server
        """
        if i_transferToServer:
            i_folder = mapServer(i_folder)

        if self.folders is None:
            self.folders = {}

        self.folders[i_name] = i_folder

    def setHost(self, value: str):
        """
        Set the server host address for this job.

        Parameters:
            - value (str): Host
        """
        self._host = value

    def setHostsMask(self, value: str):
        """
        Set the hosts mask (regex pattern) for this job.
        Hosts that match this pattern will be used to run the job.

        Parameters:
            - value (str): Hosts mask
        """
        if checkRegExp(value):
            self.hosts_mask = value

    def setHostsMaskExclude(self, value: str):
        """
        Set the hosts exclude mask (regex pattern) for this job.
        Hosts that match this pattern will not be used to run the job.

        Parameters:
            - value (str): Hosts exclude mask
        """
        if checkRegExp(value):
            self.hosts_mask_exclude = value

    def setIgnoreNimby(self):
        """
        Ignore client nimby status for this job.
        """
        self.ignorenimby = True

    def setIgnorePaused(self):
        """
        Use paused clients for this job.
        """
        self.ignorepaused = True

    def setMaintenance(self):
        """
        Set this job to be a maintenance job.
        """
        self.maintenance = True

    def setMaxRunningTasks(self, value: int):
        """
        Set the maximum number of tasks that can run simultaneously.

        Parameters:
            - value (int): Maximum number of tasks
        """
        if value >= 0:
            self.max_running_tasks = int(value)

    def setMaxRunTasksPerHost(self, value: int):
        """
        Set the maximum number of tasks that can run simultaneously on a single host.

        Parameters:
            - value (int): Maximum number of tasks
        """
        if value >= 0:
            self.max_running_tasks_per_host = int(value)

    def setName(self, name: str):
        """
        Set the name of this job.

        Parameters:
            - name (str): Name of this job
        """
        if name:
            self.name = name

    def setNativeOS(self):
        """
        Set this job to be able to run only on the OS that it was submitted from.
        """
        need_os = getVar("platform")
        if not need_os:
            platform = sys.platform
            if platform.startswith("win"):
                need_os = "windows"
            elif platform.startswith("linux"):
                need_os = "linux"
            elif platform.startswith("darwin"):
                need_os = "macosx"

        if need_os:
            self.need_os = need_os

    def setNeedOS(self, value: str):
        """
        Set the OS mask (regex pattern) for this job.

        Parameters:
            - value (str): OS mask
        """
        if checkRegExp(value):
            self.need_os = value

    def setNeedProperties(self, value: str):
        """
        Set the properties mask (regex pattern) for this job.

        Parameters:
            - value (str): Properties mask
        """
        if checkRegExp(value):
            self.need_properties = value

    def setOffline(self):
        """
        Submit this job in offline state.
        """
        self.offline = True

    def setPaused(self):
        """
        Submit this job in paused state.
        """
        self.offline = True

    def setPools(self, i_pools: list[str]):
        """
        Set the pools for this job.

        Parameters:
            - i_pools (list[str]): List of pools
        """
        self.pools = i_pools

    def setPort(self, value: int):
        """
        Set the server port for this job.

        Parameters:
            - value (str): Port
        """
        self._port = int(value)

    def setPostDeleteFiles(self, i_path: str, TransferToServer: bool = True):
        """
        Add a file (path) to be deleted after the job finishes.

        Parameters:
            - i_path (str): Path to the file to be deleted
            - TransferToServer (bool): Whether to map the path to the server
        """
        self.setCmdPost(f'deletefiles "{i_path}"', TransferToServer)

    def setPPApproval(self):
        """
        Enable post-priority approval for this job.
        """
        self.ppa = True

    def setPriority(self, priority: int):
        """
        Set this job's priority number. Has to be between 0 and 250.

        Parameters:
            - priority (int): Priority number
        """
        if priority < 0:
            return

        if priority > 250:
            priority = 250
            print(f"Warning: priority clamped to maximum = {priority}")

        self.priority = int(priority)

    def setProject(self, project: str):
        """
        Set the name of the project to which this job is related.

        Parameters:
            - project(str): Name of the project
        """
        if project:
            self.project = project

    def setTimeLife(self, value: int):
        """
        Set this job's time-life after which it will automatically be deleted.

        Parameters:
            - value (int): Life time in seconds
        """
        # This will ensure a positive integer
        if str(value).isdigit():
            self.time_life = value

    def setUserName(self, username: str):
        """
        Set the user name for the creator of this job.

        Parameters:
            - username (str): User name
        """
        if username:
            self.user_name = username.lower()

    def setWaitTime(self, value: int | float):
        """
        Set the start time for this job.

        Parameters:
            - value (int | float): Start time in seconds since epoch
        """
        if value > time.time():
            self.time_wait = int(value)

    def tryTask(self, i_block, i_task):
        """
        Add a block/task to the list of tasks to try.
        """
        if self.try_this_tasks_num is None:
            self.try_this_tasks_num = []
        if self.try_this_blocks_num is None:
            self.try_this_blocks_num = []
        self.try_this_tasks_num.append(i_task)
        self.try_this_blocks_num.append(i_block)
