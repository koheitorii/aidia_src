import json
import os
import codecs

from aidia import utils
from aidia import __version__


class LabelFileError(Exception):
    pass


class LabelFile(object):

    SUFFIX = '.json'

    def __init__(self):
        self.lf_path = None
        self.filename = None
        self.elapsed_time = 0.0
        self.brightness = 0.0
        self.contrast = 1.0
        self.note = ""
        self.shapes = None
            

    def load(self, lf_path):
        with codecs.open(lf_path, 'r', "utf-8", "ignore") as f:
            try:
                data = json.load(f)
            except Exception as e:
                return
        try:
            version = __version__
            if data.get('version') is not None:
                version = data.get('version')

            # check test or train data
            # TODO: make AI label file version format
            is_ai_file = False
            if version[0] == 'v':
                is_ai_file = True

            # get filename
            if data.get('filename') is not None:
                filename = data['filename']
                self.filename = filename

            # get elapsed time
            if data.get("elapsed_time") is not None:
                self.elapsed_time = data["elapsed_time"]

            # get brightness
            if data.get('brightness') is not None:
                brightness = data['brightness']
                self.brightness = brightness if -1.0 < brightness < 1.0 else 0.0

            # get contrast
            if data.get('contrast') is not None:
                contrast = data['contrast']
                self.contrast = contrast if 0.0 < contrast < 2.0 else 1.0

            # get note
            if data.get('note') is not None:
                note = utils.decode_note(data['note'])
                self.note = note

            # get shapes
            if data.get("shapes") is None:
                shapes = None
            elif is_ai_file:
                shapes = []
                for shape in data['shapes']:
                    n = shape['label']['number']
                    condition_dic = shape['label']['condition']
                    c = condition_dic[0]['name']
                    if c == 'MT':
                        continue
                    if c == 'N':
                        shapes.append(dict(
                            label=n,
                            shape_type=shape.get('shape_type', 'polygon'),
                            points=shape['points'],
                        ))
                    else:
                        for condition in condition_dic[1:]:
                            c = c + '_' + condition['name']
                        shapes.append(dict(
                            label=n + '_' + c,
                            shape_type=shape.get('shape_type', 'polygon'),
                            points=shape['points'],
                        ))
            else:
                shapes = [
                    dict(
                        label=shape['label'],
                        shape_type=shape.get('shape_type', 'polygon'),
                        points=shape['points'],
                    )
                    for shape in data['shapes']
                ]
            self.shapes = shapes
        except Exception as e:
            raise LabelFileError(e)
        self.lf_path = lf_path


    def save(self, lf_path, filename, height, width, elapsed_time, brightness, contrast, note, shapes):
        _note = utils.encode_note(note)
        data = dict(
            version=__version__,
            filename=filename,
            height=height,
            width=width,
            elapsed_time=elapsed_time,
            brightness=brightness,
            contrast=contrast,
            note=_note,
            shapes=shapes,
        )
        try:
            with open(lf_path, mode='w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise LabelFileError(e)
        else:
            self.lf_path = lf_path
            self.filename = filename
            self.elapsed_time = elapsed_time
            self.brightness = brightness
            self.contrast = contrast
            self.note = note
            self.shapes = shapes

    @ staticmethod
    def is_label_file(filename):
        return os.path.splitext(filename)[1].lower() == LabelFile.SUFFIX
