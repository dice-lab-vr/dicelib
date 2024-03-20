from dataclasses import dataclass
import os
import pathlib
from shutil import rmtree
from typing import List, Literal, Optional, Union

from dicelib import ui

FileType = Literal['input', 'output']
FileExtension = Literal['.tck', '.txt', '.csv']
@dataclass
class File:
    """File dataclass"""
    name: str
    type_: FileType
    path: pathlib.Path
    ext: Optional[Union[FileExtension, List[FileExtension]]] = None

    def __init__(self, name: str, type_: FileType, path: str, ext: Optional[FileExtension] = None):
        self.name = name
        self.type_ = type_
        self.path = pathlib.Path(path)
        self.ext = ext

@dataclass
class Dir:
    """Dir dataclass"""
    name: str
    path: str

Interval = Literal['()', '(]', '[)', '[]']
@dataclass
class Num:
    """Num dataclass"""
    name: str
    value: Union[int, float]
    min_: Optional[Union[int, float]] = None
    max_: Optional[Union[int, float]] = None
    include_min: Optional[bool] = True
    include_max: Optional[bool] = True
    
def check_params(files: Optional[List[File]]=None, dirs: Optional[List[Dir]]=None, nums: Optional[List[Num]]=None, force: bool=False):
    # files
    if files is not None:
        for file in files:
            if file.ext is not None:
                if isinstance(file.ext, str):
                    file.ext = [file.ext]
                if os.path.splitext(file.path)[1] not in file.ext or os.path.splitext(file.path)[1] == '':
                    exts = ' | '.join(file.ext)
                    ui.ERROR(f'Invalid extension for {file.name} file \'{file.path}\', must be {exts}')
            if file.type_ == 'input':
                if not os.path.isfile(file.path):
                    ui.ERROR(f'{file.name} file \'{file.path}\' not found')
            elif file.type_ == 'output':
                # TODO: what if output file has no extension?
                if force:
                    if os.path.isfile(file.path):
                        os.remove(file.path)
                else:
                    if os.path.isfile(file.path):
                        ui.ERROR(f'{file.name} file \'{file.path}\' already exists, use --force to overwrite')

    # dirs
    if dirs is not None:
        for dir in dirs:
            if not os.path.isdir(dir.path):
                os.mkdir(dir.path)
            else:
                if force:
                    rmtree(dir.path)
                    os.mkdir(dir.path)
                else:
                    ui.ERROR(f'{dir.name} folder \'{dir.path}\' already exists, use --force to overwrite')

    # numeric
    if nums is not None:
        for num in nums:
            if num.min_ is not None and num.max_ is not None:
                if num.include_min and num.include_max:
                    if num.value < num.min_ or num.value > num.max_:
                        ui.ERROR(f'\'{num.name}\' is not in the range ({num.min_}, {num.max_})')
                elif num.include_min and not num.include_max:
                    if num.value < num.min_ or num.value >= num.max_:
                        ui.ERROR(f'\'{num.name}\' is not in the range [{num.min_}, {num.max_})')
                elif not num.include_min and num.include_max:
                    if num.value <= num.min_ or num.value > num.max_:
                        ui.ERROR(f'\'{num.name}\' is not in the range ({num.min_}, {num.max_}]')
                elif not num.include_min and not num.include_max:
                    if num.value <= num.min_ or num.value >= num.max_:
                        ui.ERROR(f'\'{num.name}\' is not in the range [{num.min_}, {num.max_}]')
            elif num.min_ is not None:
                if num.include_min:
                    if num.value < num.min_:
                        ui.ERROR(f'\'{num.name}\' must be >= {num.min_}')
                else:
                    if num.value <= num.min_:
                        ui.ERROR(f'\'{num.name}\' must be > {num.min_}')
            elif num.max_ is not None:
                if num.include_max:
                    if num.value > num.max_:
                        ui.ERROR(f'\'{num.name}\' must be <= {num.max_}')
                else:
                    if num.value >= num.max_:
                        ui.ERROR(f'\'{num.name}\' must be < {num.max_}')
