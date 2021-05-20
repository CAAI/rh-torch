# -*- coding: utf-8 -*-
"""
Created on Thu May 20 19:31:52 2021

@author: clad0003
"""

import numpy as np
from typing import Union, Optional
#import pathlib
from pathlib import Path

def cutup(data: np.ndarray, blck: tuple[int,...], strd: tuple[int,...]) -> np.ndarray:
    """
    

    Parameters
    ----------
    data : np.ndarray
        The input array (2d,3d,...) from which to extract patches.
    blck : tuple[int,...]
        The size of the patch to be extracted.
    strd : tuple[int,...]
        The stride to move the patch.

    Returns
    -------
    patches : np.ndarray
        The extracted patches, shaped as two times the blck.shape.
        For a 2D array, e.g.: (ind_i,ind_j,*blck),where ind_i and ind_j is 
        the number of patches in the first and second direction, respectively. 
    """
    from numpy.lib import stride_tricks
    
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    patches = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return patches

class PatchesBuilder:
    """ Helper class to build patches dynamically and fast 
    Outputs:
        
        Channels last: 
    """
    def __init__(self,channels_first: bool=True):
        self.data = []
        self.channels_first = channels_first
        
    def update(self, row):
        for r in row:
            self.data.append(r)

    def finalize(self) -> np.ndarray:
        patches = np.reshape(self.data, newshape=(len(self.data), *self.data[0].shape))
        # (color_channel,num_patches,*patch_size)
        if not self.channels_first:
            # (num_patches,*patch_size,color_channel)
            patches = np.moveaxis(patches,0,-1)
        return patches
    
def save_patch( patch, path, save_memmap ):
    if save_memmap:
        pass
        # TO BE IMPLEMENTED
    else:
        np.save( path , patch )
     

def generate_patches_from_ndarray(identifier: str,
                                  inputs: Union[np.ndarray,list],
                                  outputs: Union[np.ndarray,list],
                                  patch_size: tuple[int,...],
                                  stride: tuple[int,...],
                                  out_dir: Union[Path,str], 
                                  out_inputs_dir: Optional[str]=None,
                                  out_inputs_prefix: Optional[str]='',
                                  out_outputs_dir: Optional[str]=None,
                                  out_outputs_prefix: Optional[str]='',
                                  channels_first: bool=True,
                                  save_memmap: bool=False):
    
    input_patches = PatchesBuilder(channels_first)
    output_patches = PatchesBuilder(channels_first)
    
    
    if not isinstance(inputs,list): inputs = [inputs]
    if not isinstance(outputs,list): outputs = [outputs]
    
    """ Generate the arrays of patches """
    for input_volume in inputs:
        patches = cutup(input_volume,patch_size,stride)
        ijk = np.prod(patches.shape[:len(patch_size)])
        input_patches.update( np.reshape(patches,(ijk,*patch_size)) )
    input_patches = input_patches.finalize()
    
    for output_volume in outputs:
        patches = cutup(output_volume,patch_size,stride)
        output_patches.update( np.reshape(patches,(ijk,*patch_size)) )
    output_patches = output_patches.finalize()
    
    
    """ Set up save path """
    # <output_directory>/
    out_dir = Path(out_dir) if isinstance(out_dir,str) else out_dir
    
    # If set, add out_<inputs/outputs>_dir to path ( e.g. <output_directory>/patches/ )
    input_out_path = out_dir.joinpath(out_inputs_dir) if out_inputs_dir is not None else out_dir
    output_out_path = out_dir.joinpath(out_outputs_dir) if out_outputs_dir is not None else out_dir
    input_out_path.mkdir(parents=True,exist_ok=True)
    output_out_path.mkdir(parents=True,exist_ok=True)
    
    # Build prefix of patch name
    patch_name = '{prefix}{identifier}{patch_id}.npy'
    if not out_inputs_prefix == '' and not out_inputs_prefix.endswith('_'):
        out_inputs_prefix += '_'
    if not out_outputs_prefix == '' and not out_outputs_prefix.endswith('_'):
        out_outputs_prefix += '_'
    
    """ Extract and save the patches """    
    patch_index_axis = 1 if channels_first else 0
    for ind in range(ijk):
        
        # Todo: Skip patches that doesnt parse a check, 
        # e.g. check for minimum max-value in either patch
        # Should be supplied as a function pointer
        
        # Save inputs patch to file
        save_patch( input_out_path.joinpath( patch_name.format( prefix=out_inputs_prefix, 
                                                                identifier=identifier,
                                                                patch_id=ind) ), 
                    input_patches.take(indices=ind,axis=patch_index_axis),
                    save_memmap)
        
        # Save outputs patch to file
        save_patch( output_out_path.joinpath( patch_name.format( prefix=out_outputs_prefix, 
                                                                 identifier=identifier,
                                                                 patch_id=ind) ), 
                    output_patches.take(indices=ind,axis=patch_index_axis),
                    save_memmap)
