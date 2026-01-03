class FRED_Multi_Linear_Slider:
    """
    Master Slider from 0–100 to control 5 Boundary limits output in INT or FLOAT.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_slider": ("FLOAT", { 
                    "display": "slider", 
                    "default": 50.0, 
                    "min": 0.0, 
                    "max": 100.0, 
                    "step": 0.1, 
                    "tooltip": "Master slider (0–100)."
                }),

                # Boundary of value 1
                "Boundary_1_MIN": ("FLOAT", {"default": 0.0, "min": -999999, "max": 999999, "step": 0.1}),
                "Boundary_1_MAX": ("FLOAT", {"default": 1.0, "min": -999999, "max": 999999, "step": 0.1}),

                # Boundary of value 2
                "Boundary_2_MIN": ("FLOAT", {"default": 0.0, "min": -999999, "max": 999999, "step": 0.1}),
                "Boundary_2_MAX": ("FLOAT", {"default": 1.0, "min": -999999, "max": 999999, "step": 0.1}),
                
                # Boundary of value 3
                "Boundary_3_MIN": ("FLOAT", {"default": 0.0, "min": -999999, "max": 999999, "step": 0.1}),
                "Boundary_3_MAX": ("FLOAT", {"default": 1.0, "min": -999999, "max": 999999, "step": 0.1}),
                
                # Boundary of value 4
                "Boundary_4_MIN": ("FLOAT", {"default": 0.0, "min": -999999, "max": 999999, "step": 0.1}),
                "Boundary_4_MAX": ("FLOAT", {"default": 1.0, "min": -999999, "max": 999999, "step": 0.1}),
                
                # Boundary of value 5
                "Boundary_5_MIN": ("FLOAT", {"default": 0.0, "min": -999999, "max": 999999, "step": 0.1}),
                "Boundary_5_MAX": ("FLOAT", {"default": 1.0, "min": -999999, "max": 999999, "step": 0.1}),
            }
        }

    RETURN_TYPES = (
        "FLOAT", "INT",
        "FLOAT", "INT",
        "FLOAT", "INT",
        "FLOAT", "INT",
        "FLOAT", "INT",
    )

    RETURN_NAMES = (
        "value1_float", "value1_int",
        "value2_float", "value2_int",
        "value3_float", "value3_int",
        "value4_float", "value4_int",
        "value5_float", "value5_int",
    )

    FUNCTION = "calculate"
    CATEGORY = "👑FRED/utils"
    DESCRIPTION = "Master 0–100 slider mapped to 5 linear values between custom boundary limits (FLOAT + INT rounded)."

    def _map_linear(self, master, boundary_min, boundary_max):
        t = max(0.0, min(100.0, float(master))) / 100.0
        value_float = boundary_min + (boundary_max - boundary_min) * t
        value_int = int(round(value_float))
        return value_float, value_int

    def calculate(
        self,
        master_slider,
        Boundary_1_MIN, Boundary_1_MAX,
        Boundary_2_MIN, Boundary_2_MAX,
        Boundary_3_MIN, Boundary_3_MAX,
        Boundary_4_MIN, Boundary_4_MAX,
        Boundary_5_MIN, Boundary_5_MAX,
    ):
        v1_float, v1_int = self._map_linear(master_slider, Boundary_1_MIN, Boundary_1_MAX)
        v2_float, v2_int = self._map_linear(master_slider, Boundary_2_MIN, Boundary_2_MAX)
        v3_float, v3_int = self._map_linear(master_slider, Boundary_3_MIN, Boundary_3_MAX)
        v4_float, v4_int = self._map_linear(master_slider, Boundary_4_MIN, Boundary_4_MAX)
        v5_float, v5_int = self._map_linear(master_slider, Boundary_5_MIN, Boundary_5_MAX)

        return (
            v1_float, v1_int,
            v2_float, v2_int,
            v3_float, v3_int,
            v4_float, v4_int,
            v5_float, v5_int,
        )


NODE_CLASS_MAPPINGS = {
    "FRED_Multi_Linear_Slider": FRED_Multi_Linear_Slider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_Multi_Linear_Slider": "👑 FRED Multi Linear Slider",
}
