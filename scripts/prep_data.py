import rdkit.Chem as Chem
import numpy as np
from tqdm import tqdm

'''
This script prepares the data used in Wengong Jin's NIPS paper on predicting reaction outcomes for the modified
forward prediction script. Rather than just training to predict which bonds change, we make a direct prediction
on HOW those bonds change
'''

def get_changed_bonds(rxn_smi):
    reactants = Chem.MolFromSmiles(rxn_smi.split('>')[0])
    products  = Chem.MolFromSmiles(rxn_smi.split('>')[2])

    conserved_maps = [a.GetProp('molAtomMapNumber') for a in products.GetAtoms() if a.HasProp('molAtomMapNumber')]
    bond_changes = set() # keep track of bond changes

    # Look at changed bonds
    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps): continue
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()


    for bond in bonds_prev:
        if bond not in bonds_new:
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], 0.0)) # lost bond
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond])) # changed bond
    for bond in bonds_new:
        if bond not in bonds_prev:
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))  # new bond

    return bond_changes


def process_file(fpath):
    with open(fpath, 'r') as fid_in, open(fpath + '.proc_fulldata', 'w') as fid_out:
        i = 0
        for line in tqdm(fid_in):
            rxn_smi = line.strip().split(' ')[0]
            bond_changes = get_changed_bonds(rxn_smi)
            fid_out.write('{} {}\n'.format(rxn_smi, ';'.join(['{}-{}-{}'.format(x[0], x[1], x[2]) for x in bond_changes])))
    print('Finished processing {}'.format(fpath))

if __name__ == '__main__':

    # Process files
    # process_file('../data/1mapped_num.txt')
    # process_file('../data/2mapped_num.txt')
    # process_file('../data/3mapped_num.txt')
    # process_file('../data/4mapped_num.txt')

    process_file('../data/train.txt')
    process_file('../data/valid.txt')
    process_file('../data/test.txt')
    process_file('../data/test_human.txt')