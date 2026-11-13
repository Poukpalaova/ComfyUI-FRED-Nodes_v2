# FRED_TextReplaceList.py
# Node that checks prompt and replaces words from lists with specified replacements
# Similar to CR_TextReplace but with list support and on/off toggles

class FRED_TextReplaceList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            },
            "optional": {
                # Group 1
                "enable1": ("BOOLEAN", {"default": True}),
                "find_list1": ("STRING", {"multiline": True, "default": "", "placeholder": "word1, word2, word3"}),
                "replace1": ("STRING", {"multiline": False, "default": ""}),
                # Group 2
                "enable2": ("BOOLEAN", {"default": False}),
                "find_list2": ("STRING", {"multiline": True, "default": "", "placeholder": "word1, word2, word3"}),
                "replace2": ("STRING", {"multiline": False, "default": ""}),
                # Group 3
                "enable3": ("BOOLEAN", {"default": False}),
                "find_list3": ("STRING", {"multiline": True, "default": "", "placeholder": "word1, word2, word3"}),
                "replace3": ("STRING", {"multiline": False, "default": ""}),
                # Group 4
                "enable4": ("BOOLEAN", {"default": False}),
                "find_list4": ("STRING", {"multiline": True, "default": "", "placeholder": "word1, word2, word3"}),
                "replace4": ("STRING", {"multiline": False, "default": ""}),
                # Group 5
                "enable5": ("BOOLEAN", {"default": False}),
                "find_list5": ("STRING", {"multiline": True, "default": "", "placeholder": "word1, word2, word3"}),
                "replace5": ("STRING", {"multiline": False, "default": ""}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "replace_text_lists"
    CATEGORY = "FRED/Utils"
    
    def replace_text_lists(self, text, 
                           enable1=False, find_list1="", replace1="",
                           enable2=False, find_list2="", replace2="",
                           enable3=False, find_list3="", replace3="",
                           enable4=False, find_list4="", replace4="",
                           enable5=False, find_list5="", replace5=""):
        
        text_out = text
        
        # Process each group if enabled
        if enable1 and find_list1.strip():
            text_out = self._replace_from_list(text_out, find_list1, replace1)
        
        if enable2 and find_list2.strip():
            text_out = self._replace_from_list(text_out, find_list2, replace2)
        
        if enable3 and find_list3.strip():
            text_out = self._replace_from_list(text_out, find_list3, replace3)
        
        if enable4 and find_list4.strip():
            text_out = self._replace_from_list(text_out, find_list4, replace4)
        
        if enable5 and find_list5.strip():
            text_out = self._replace_from_list(text_out, find_list5, replace5)
        
        return (text_out,)
    
    def _replace_from_list(self, text, find_list_str, replace_str):
        """Replace all words from a comma-separated list with the replacement string"""
        # Split by comma and strip whitespace
        words_to_replace = [word.strip() for word in find_list_str.split(",") if word.strip()]
        
        for word in words_to_replace:
            if word:  # Make sure word is not empty
                text = text.replace(word, replace_str)
        
        return text


# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_TextReplaceList": FRED_TextReplaceList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_TextReplaceList": "👑 FRED_TextReplaceList"
}