from fire import Fire
import os
from ase.io import read, write
import numpy as np
import itertools
from typing import Union
from glob import glob
import random as r


def convert_xyz_to_sys_data(in_file: str, out_dir: str):
    atoms = read(in_file, "::")
    atoms = r.sample(atoms, 20)
    for i, atom in enumerate(atoms):
        out_file = os.path.join(out_dir, "POSCAR%03d" % i)
        write(out_file, atom, sort=True)


def add_xyz_to_init_data(in_files: Union[str, list], data_set_dir: str, atoms_kind=0):

    if isinstance(in_files, str):
        in_files = [in_files]
    elif not isinstance(in_files, list):
        raise ValueError('in_files should be string or list of string')

    file_paths = set()
    for file_path in in_files:
        file_paths.update(glob(file_path, recursive=True))

    idx_iter = itertools.count(start=0, step=1)

    for file_path in sorted(file_paths):
        data_set_path = None
        for idx in idx_iter:
            set_name = 'set.{}'.format(str(idx).zfill(3))
            data_set_path = os.path.join(data_set_dir, set_name)
            try:
                os.makedirs(data_set_path, exist_ok=False)
                break
            except FileExistsError as e:
                pass
        convert_xyz_to_init_data(file_path, data_set_path, atoms_kind)
        print('convert and add {} to training data set {} successfully!'.format(
            file_path, data_set_path))


def convert_xyz_to_init_data(in_file: str, out_dir: str, atoms_kind=0):
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

    for name, data in (('force', force), ('energy', energy), ('coord', coord), ('box', box)):
        out_file = '{}.npy'.format(name)
        out_file_path = os.path.join(out_dir, out_file)
        np.save(out_file_path, data)
    raw_file_path = os.path.join(out_dir, 'type.raw')
    with open(raw_file_path, 'w') as f:
        f.write(' '.join(type_raw))


if __name__ == '__main__':
    Fire(dict(
        convert_xyz_to_init_data=convert_xyz_to_init_data,
        convert_xyz_to_sys_data=convert_xyz_to_sys_data,
        add_xyz_to_init_data=add_xyz_to_init_data,
    ))
