from fire import Fire
import os
from ase.io import read
import numpy as np


def xyz_to_dp_train_data(in_file: str, output_dir: str, atoms_kind=0):
    os.makedirs(output_dir, exist_ok=True)
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

    for name, data in [('force', force), ('energy', energy), ('coord', coord), ('box', box)]:
        out_file = '{}.npy'.format(name)
        out_file_path = os.path.join(output_dir, out_file)
        np.save(out_file_path, data)

    raw_file_path = os.path.join(output_dir, 'type.raw')
    with open(raw_file_path, 'w') as f:
        f.write(' '.join(type_raw))


if __name__ == '__main__':
    Fire(dict(
        xyz_to_dp_train_data=xyz_to_dp_train_data,
    ))
