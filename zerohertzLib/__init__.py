from zerohertzLib import algorithm, api, logging, mlops, monitoring, plot

try:
    from zerohertzLib import vision
except ImportError as e:
    print("=" * 100)
    print(f"[Warning] {e}")
    print("Please Install OpenCV Dependency")
    print("--->\t$ sudo apt install python3-opencv -y\t<---")
    print("(but you can use other submodules except zerohertzLib.vision)")
    print("=" * 100)

__version__ = "v0.2.3"
