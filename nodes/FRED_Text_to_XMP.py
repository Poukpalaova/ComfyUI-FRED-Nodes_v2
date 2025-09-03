import re

# Expanded English stopword list (can be further customized)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "of", "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "over", "under",
    "and", "but", "or", "nor", "so", "yet",
    "very", "just", "only", "also", "even", "still", "such", "no", "not",
    "too", "than", "then", "once", "here", "there", "when", "where", "why", "how",
    "this", "that", "these", "those", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "own", "same",
}

HELP_MESSAGE = """
👑 FRED_Text_to_XMP

🔹 PURPOSE:
Create and attach XMP metadata to your images (or write a sidecar .xmp) from free text or structured fields.
Useful to archive prompts/parameters/keywords in a standard, app-friendly format (Lightroom, Bridge, Photoshop, digiKam, DAMs).

📥 INPUTS:
- image • input image to annotate (pass-through if you only want a sidecar)
- xmp_body • the text to convert into XMP. Accepts:
  • Plain "key = value" lines
  • JSON (object) for complex fields
  • Namespaced keys (dc:subject, xmp:Rating, photoshop:AuthorsPosition, exif:UserComment, etc.)
- include_prompt / include_negative / include_parameters • auto-inject these strings into XMP (if provided upstream)
- keywords • comma-separated list → stored as dc:subject (Bag)
- creator / title / description / rights • common Dublin Core/Photoshop fields
- rating • 0–5 (xmp:Rating)
- write_mode • "embed_png", "embed_jpeg", "embed_tiff", "sidecar_only" (what to write and where)
- sidecar_path • template/path for .xmp when sidecar_only or when embedding is unavailable
- merge_strategy • "merge", "overwrite", "skip" (how to resolve existing XMP in the input)
- pretty_print • human-readable XML
- sanitize_xml • escape unsafe characters (& < > ") in values
- time_format • used for tokens inside paths/fields
- token context • optional dict to expand %tokens (model/prompt/seed/etc.)

⚙️ KEY OPTIONS:
- Namespaces & mapping:
  • dc:title / dc:description / dc:subject (keywords as rdf:Bag)
  • xmp:Rating (0–5), xmp:CreateDate / xmp:MetadataDate (UTC ISO)
  • photoshop:AuthorsPosition / photoshop:Headline (optional)
  • exif:UserComment for prompt/notes if you want EXIF readers to see it
- Input parsing:
  • Line mode: one "key = value" per line → trims spaces; empty lines ignored.
  • JSON mode: if xmp_body starts with '{', it’s parsed as JSON (nested objects/arrays allowed).
  • Arrays: comma-separated values map to rdf:Bag by default for dc:* fields.
- Auto-fill (if toggled):
  • include_prompt → writes to exif:UserComment and dc:description
  • include_negative → appends to dc:description with a "Negative:" section
  • include_parameters → writes a compact block into xmpMM:History or exif:UserComment
- Tokens in fields and paths:
  • %date, %date_dash, %time, %datetime
  • %model_name, %seed, %steps, %width, %height, %guidance/%cfg, %denoise, etc.
  • %img_count for batches (if provided)
- Embedding vs. sidecar:
  • PNG/TIFF: embeds XMP packet in the file (when supported by the format/writer)
  • JPEG: embeds standard XMP segment
  • Sidecar: writes adjacent .xmp XML file (good for non-destructive, DAM-friendly workflows)
- Safety:
  • sanitize_xml prevents malformed XML; long values are preserved (no hard truncation)
  • merge_strategy decides how to handle pre-existing XMP in the input image

📤 OUTPUTS:
- IMAGE • pass-through image (unchanged pixels)
- XMP_STRING • the final XMP packet text (UTF-8)
- SIDE_CAR_PATH • full path to the written .xmp (empty if none)
- help • this message

📝 Tips:
- Prefer dc:subject (keywords) for searchability in DAM/Lightroom.
- Keep prompts in exif:UserComment + dc:description; readers vary by app.
- Sidecar only is safest when you don’t want to modify the original image file.
- Use tokens to stamp model/version/seed for reliable provenance.
"""


class FRED_Text_to_XMP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING",),
                "sentence_mode": ("BOOLEAN", {"default": False}),
                "replace_space_with_underscore": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("XMP_formatted_text", "help",)
    CATEGORY = "👑FRED/utils"
    FUNCTION = "convert_to_xmp"

    def convert_to_xmp(self, text: str, sentence_mode: bool, replace_space_with_underscore: bool):
        tags = self.extract_tags(text, sentence_mode, replace_space_with_underscore)
        xmp = self.tags_to_xmp(tags)
        return (xmp, HELP_MESSAGE)

    def extract_tags(self, text, sentence_mode, replace_space):
        if sentence_mode:
            # Remove punctuation, split into words, filter stopwords
            words = re.findall(r'\b\w+\b', text.lower())
            tags = [w for w in words if w not in STOPWORDS]
        else:
            # Split by comma, strip spaces
            tags = [t.strip() for t in text.split(",") if t.strip()]
        if replace_space:
            tags = [t.replace(" ", "_") for t in tags]
        return tags

    def tags_to_xmp(self, tags):
        li_elements = "\n".join(f"          <rdf:li>{tag}</rdf:li>" for tag in tags)
        return f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description xmlns:dc='http://purl.org/dc/elements/1.1/'>
      <dc:subject>
        <rdf:Bag>
{li_elements}
        </rdf:Bag>
      </dc:subject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FRED_Text_to_XMP": FRED_Text_to_XMP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_Text_to_XMP": "👑 FRED_Text_to_XMP"
}