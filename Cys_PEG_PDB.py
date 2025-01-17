from Bio.PDB import PDBParser, PDBIO, Model, Chain, Residue, Atom
import numpy as np

# Paths to the PDB files
cdc20_pdb_path = '/content/AF-Q9GZT9-F1-model_v4.pdb'
peg_pdb_path = '/content/F0vmOQ.pdb'

# Initialize the PDB parser
parser = PDBParser(QUIET=True)

# Parse the PDB files
cdc20_structure = parser.get_structure('cdc20', cdc20_pdb_path)
peg_structure = parser.get_structure('PEG', peg_pdb_path)

# Get the PEG molecule (assuming it is the first and only chain in the PEG structure)
peg_chain = next(peg_structure.get_chains())

# Identify cysteine residues in the cdc20 structure
cysteine_residues = [residue for residue in cdc20_structure.get_residues() if residue.get_resname() == 'CYS']

# Create a new chain to hold the PEG-modified residues
new_chain = Chain.Chain('P')  # 'P' for PEG

# Iterate over each cysteine residue and attach the PEG molecule to the SG atom
for i, cysteine in enumerate(cysteine_residues):
    sg_atom = cysteine['SG']
    
    # Translate the PEG molecule so that its first atom overlaps with the SG atom
    peg_atoms = list(peg_chain.get_atoms())
    translation_vector = sg_atom.coord - peg_atoms[0].coord
    
    # Create a new residue to hold the PEG atoms
    new_residue = Residue.Residue(('H_PEG', i + 1, ' '), 'PEG', i + 1)
    
    for atom in peg_atoms:
        # Create a new atom with the translated coordinates
        new_atom = Atom.Atom(name=atom.name,
                             coord=atom.coord + translation_vector,
                             bfactor=atom.bfactor,
                             occupancy=atom.occupancy,
                             altloc=atom.altloc,
                             fullname=atom.fullname,
                             serial_number=atom.serial_number,
                             element=atom.element)
        
        # Add the new atom to the new residue
        new_residue.add(new_atom)
    
    # Add the new residue to the new chain
    new_chain.add(new_residue)

# Add the new chain to the model (assuming only one model exists)
cdc20_structure[0].add(new_chain)

# Save the modified structure to a new PDB file
io = PDBIO()
io.set_structure(cdc20_structure)
io.save('PEG_modified.pdb')

print("PEG has been attached to each cysteine residue and saved as 'PEG_modified.pdb'.")
