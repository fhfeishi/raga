# patent_parser.py

from pydantic import BaseModel
from collections import OrderedDict 
from typing import Dict, Tuple, List , Any
from pathlib import Path 

class PatentParser:
    def __init__(self, mdf: str) -> Any:
        self.mdf : Path = Path(mdf) 
        self.fdir: Path = self.mdf.parent.resolve()
        self.imdir: Path = self.fdir / "images"
        assert self.imdir.is_dir() is True
        self.pdff: Path = next(self.fdir.rglob('*_origon.pdf'), None)
        assert self.pdff.is_file() 
        self.text = ""
        
        
        
        
        


