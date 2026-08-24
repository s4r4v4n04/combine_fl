import subprocess
import math
import psutil

from utils.logger import FedLogger

logger = FedLogger("0", "HARDWARE_INFO")


def get_hardware_info():
    hardware_info = dict()
    try:
        output = subprocess.check_output(
            [
                "lscpu",
            ]
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"hardware_info.get_hardware_info.exception:: Failed to get lscpu info: {e}")
        logger.error("hardware_info.get_hardware_info.exception", f"Failed to get lscpu info: {e}")
        output = None
    except Exception as e:
        print(f"hardware_info.get_hardware_info.unknown_exception:: Failed to get lscpu info: {e}")
        logger.error("hardware_info.get_hardware_info.unknown_exception", f"Failed to get lscpu info: {e}")
        output = None

    if output:
        output = output.decode("utf-8").strip().strip().split("\n")
        for line in output:
            line = line.split(":")
            if "Architecture" in line[0]:
                hardware_info["arch"] = line[1].strip()
            elif "CPU(s)" == line[0]:
                hardware_info["cpu_core_count"] = line[1].strip()
            elif "Model name" in line[0]:
                hardware_info["model_name"] = line[1].strip()
    else:
        hardware_info = {"arch": None, "cpu_core_count": None, "model_name": None}

    try:
        import torch
        if torch.cuda.is_available():
            hardware_info["cuda_available"] = True
            # Get VRAM in GB
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            hardware_info["vram_gb"] = math.ceil(vram_bytes / (1024**3))
        else:
            hardware_info["cuda_available"] = False
            hardware_info["vram_gb"] = 0
    except Exception as e:
        print(f"hardware_info.get_hardware_info.exception:: Failed to get VRAM: {e}")
        hardware_info["cuda_available"] = False
        hardware_info["vram_gb"] = 0

    try:
        import psutil
        # Get total RAM in Bytes, then convert to Gigabytes (GB) and ceiling
        total_ram_bytes = psutil.virtual_memory().total
        hardware_info["total_ram_gb"] = math.ceil(total_ram_bytes / (1024**3))
    except Exception as e:
        logger.error("hardware_info.get_hardware_info.exception", f"Failed to get RAM info: {e}")
        hardware_info["total_ram_gb"] = "Unknown"

    return hardware_info


if __name__ == "__main__":
    print(get_hardware_info())
