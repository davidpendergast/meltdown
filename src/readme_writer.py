import re
import traceback
import os


def write_basic_readme(src_file="README_template.txt", dest_file="README.md", gif_dir="gifs"):
    gif_filenames = [f for f in os.listdir(gif_dir) if os.path.isfile(os.path.join(gif_dir, f))]
    gif_filenames = [f for f in gif_filenames if f.endswith(".gif") and f[0].isdigit()]
    gif_filenames.sort(key=lambda text: parse_leading_int(text, or_else=-1), reverse=True)

    def _key_lookup(key: str):
        n = parse_ending_int(key, or_else=-1)
        if n < 0 or n >= len(gif_filenames):
            return None
        if key.startswith("file_"):
            return gif_filenames[n]
        elif key.startswith("name_"):
            return gif_filenames[n][:-4]  # rm the ".gif" part
        else:
            return None

    write_readme(src_file, dest_file,
                 key_lookup=_key_lookup,
                 skip_line_if_value_missing=True)


def parse_leading_int(text, or_else=-1):
    just_the_num = re.search(r"\d+", text).group()
    if just_the_num == "":
        return or_else
    else:
        return int(just_the_num)


def parse_ending_int(text, or_else=-1):
    just_the_num = re.search(r"\d+$", text).group()
    if just_the_num == "":
        return or_else
    else:
        return int(just_the_num)


def write_readme(template_file, dest_file, key_lookup=lambda key: None, skip_line_if_value_missing=True):
    try:
        print("INFO: updating {}".format(dest_file))
        with open(template_file, "r") as f:
            template_lines = f.readlines()

        result_lines = []
        for line in template_lines:
            line = _process_line(line, key_lookup, skip_line_if_value_missing)
            if line is not None:
                result_lines.append(line)

        with open(dest_file, "w") as dest_f:
            dest_f.write("".join(result_lines))

    except Exception as e:
        traceback.print_exc()
        print("ERROR: failed to generate readme")


def _process_line(line, key_lookup, skip_line_if_value_missing):
    all_keys = re.findall("{[^}]*}", line)  # finds anything like "{this}"
    for key in all_keys:
        key_text = key[1:len(key) - 1]  # remove brackets
        new_value = key_lookup(key_text)
        if new_value is not None:
            # XXX if you decide to replace a key with another key, you'll have issues here
            line = line.replace(key, str(new_value), 1)
        else:
            if skip_line_if_value_missing:
                return None
    return line
