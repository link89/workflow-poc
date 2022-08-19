from audioop import add
from fire import Fire
import os
from ase.io import read
import numpy as np
import itertools


def add_xyz_to_training_data_set(in_file: str, out_dir: str, set_id: int = None, atoms_kind=0):
    ats = read(in_file, ':')

    force = np.array([np.ravel(at.get_forces()) for at in ats])
    coord = np.array([np.ravel(at.get_positions()) for at in ats])
    energy = np.array([at.get_potential_energy() for at in ats])
    box = [at.get_cell().reshape(9) for at in ats]

    if atoms_kind == 0:
        symbol_set = set(ats[0].get_chemical_symbols())
    else:
        symbol_set = atoms_kind
    sym_dict = dict(zip(symbol_set, range(len(symbol_set))))
    type_raw = [str(sym_dict[specie])
                for specie in ats[0].get_chemical_symbols()]
        
    train_data_set_path = None
    
    for idx in (itertools.count(start=0, step=1) if set_id is None else [set_id]):
        set_name = 'set.{}'.format(str(idx).zfill(3))
        train_data_set_path = os.path.join(out_dir, set_name)
        try:
            os.makedirs(train_data_set_path, exist_ok=False)
            break
        except FileExistsError as e :
            if set_id is not None:
                raise
    
    for name, data in (('force', force), ('energy', energy), ('coord', coord), ('box', box)):
        out_file = '{}.npy'.format(name)
        out_file_path = os.path.join(train_data_set_path, out_file)
        np.save(out_file_path, data)

    raw_file_path = os.path.join(out_dir, 'type.raw')
    with open(raw_file_path, 'w') as f:
        f.write(' '.join(type_raw))

if __name__ == '__main__':
    Fire(dict(
        add_xyz_to_training_data_set=add_xyz_to_training_data_set,
    ))
