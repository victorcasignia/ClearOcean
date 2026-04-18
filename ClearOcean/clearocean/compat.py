import sys
import types


def patch_torchvision_compat():
    """Patch missing torchvision.transforms.functional_tensor for basicsr compatibility."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    try:
        from torchvision.transforms import functional as functional
    except Exception:
        return

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    if hasattr(functional, "rgb_to_grayscale"):
        shim.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim
