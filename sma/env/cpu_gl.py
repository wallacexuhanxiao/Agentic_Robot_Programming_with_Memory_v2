"""CPU-only rendering for jobs without a GPU: robosuite's EGL context insists on an EGL device display, which a node
without a GPU does not offer. Mesa's surfaceless platform (software rasterizer) gives a display with no device; MuJoCo
renders into its own offscreen framebuffer, so no window surface is needed."""
EGL_PLATFORM_SURFACELESS_MESA = 0x31DD


def enable_surfaceless():
    import robosuite.renderers.context.egl_context as context
    from mujoco.egl import egl_ext as EGL

    def surfaceless_display(device_id=0):
        native = getattr(EGL, "EGL_DEFAULT_DISPLAY", None)
        display = EGL.eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, native, None)
        if display == EGL.EGL_NO_DISPLAY:
            return EGL.EGL_NO_DISPLAY
        return display if EGL.eglInitialize(display, None, None) == EGL.EGL_TRUE else EGL.EGL_NO_DISPLAY

    context.create_initialized_egl_device_display = surfaceless_display
