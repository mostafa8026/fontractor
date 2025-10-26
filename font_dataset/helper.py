from fontTools.ttLib import TTFont


__all__ = ["char_in_font"]


def char_in_font(unicode_char, font_path):
    try:
        # print(
        #     f"Checking if character '{unicode_char}' '{ord(unicode_char)}' is in font '{font_path}'")
        font = TTFont(font_path, fontNumber=0)
        # print font["cmap"].tables [<fontTools.ttLib.tables._c_m_a_p.cmap_format_4 object at 0x745c8d4f9930>]
        # for cmap in font["cmap"].tables:
        #     print(cmap.cmap)
        for cmap in font["cmap"].tables:
            if cmap.isUnicode():
                if ord(unicode_char) in cmap.cmap:
                    return True
        return False
    except Exception as e:
        return False
