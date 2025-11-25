import os
import base64
import unicodedata
import openpyxl


def is_full_width(text):
    for ch in text:
        if unicodedata.east_asian_width(ch) in 'FWA':
            return True
    return False

def join(s1, s2, sep="/"):
    s1 = s1.replace(os.sep, sep)
    s2 = s2.replace(os.sep, sep).strip(sep)
    if s1[-1] != sep:
        s1 = s1 + sep
    return s1 + s2

def get_parent_path(path):
    return os.path.abspath(os.path.join(path, os.pardir))

def encode_note(data):
    return base64.b64encode(data.encode()).decode('utf-8')

def decode_note(data):
    return base64.b64decode(data.encode('utf-8')).decode()

def target_in_list(target: list, l: list):
    if target is None:
        return False
    
    for t in target:
        if t in l:
            pass
        else:
            return False
    return True

def ravel_lists(l: list):
    return list(set(sum(l, [])))

def extract_ext(path):
    return os.path.splitext(os.path.basename(path))[1].lower()

def get_basename(path):
        return os.path.splitext(os.path.basename(path))[0]

def get_basedir(path):
    return os.path.basename(os.path.dirname(path))

def save_dict_to_excel(data, file_path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    headers = list(data.keys())
    sheet.append(headers)

    for row_data in zip(*[data[header] for header in headers]):
        sheet.append(row_data)

    workbook.save(file_path)

def get_dirpath_with_mkdir(*args):
    p = os.path.join(*args)
    if not os.path.exists(p):
        os.makedirs(p)
    return p