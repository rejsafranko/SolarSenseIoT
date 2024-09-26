import os
import sys
from dotenv import load_dotenv

load_dotenv()
REQUIRED_PYTHON = os.getenv("REQUIRED_PYTHON")


def main():
    system_version = sys.version.replace("\n", "")

    if REQUIRED_PYTHON.startswith("python"):
        required_version_str = REQUIRED_PYTHON[len("python") :].strip()
        required_version_parts = required_version_str.split(".")
        if len(required_version_parts) != 3:
            raise ValueError(
                "Invalid required Python version format: {}".format(REQUIRED_PYTHON)
            )

        required_major = int(required_version_parts[0])
        required_minor = int(required_version_parts[1])
        required_micro = int(required_version_parts[2])
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(REQUIRED_PYTHON))

    system_major = sys.version_info.major
    system_minor = sys.version_info.minor
    system_micro = sys.version_info.micro

    if (system_major, system_minor, system_micro) != (
        required_major,
        required_minor,
        required_micro,
    ):
        raise TypeError(
            "This project requires Python {}.{}.{}. Found: Python {}".format(
                required_major, required_minor, required_micro, system_version
            )
        )

    print(">>> Development environment passes all tests!")


if __name__ == "__main__":
    main()
