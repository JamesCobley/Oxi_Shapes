# Oxi_Shapes
This repository contains the [Python](https://www.python.org/)-scripted source code for the preprint entitled "Oxi-Shapes: The Theroem-derived Proteoform Geometry". The open-sourced code is deisgned to be used in the freely-available Google [Colab](https://colab.google/) environment. 
# Scripts
A breif overview of the scripts by their use-case is as follows:
1. Figure_1. Composed of panels derived from [Fig_script_1](https://github.com/JamesCobley/Oxi_Shapes/blob/main/Fig_script_1.py) ,[Fig_script_2](https://github.com/JamesCobley/Oxi_Shapes/blob/main/Fig_script_1.py) , and [Fig_script_3](https://github.com/JamesCobley/Oxi_Shapes/blob/main/Fig_script_3.py). Extended_Data_Figure_1 was created from elements of the aforementioned scripts and the [ExF1](https://github.com/JamesCobley/Oxi_Shapes/blob/main/ExF1.py) script. Extended data Figure 2 was created [ExF2](https://github.com/JamesCobley/Oxi_Shapes/blob/main/ExF2.py) script.
2. Figure_2. The flat and curved Oxi-Shapes were initially derived using the static scripts (e.g., [R4_oxi](https://github.com/JamesCobley/Oxi_Shapes/blob/main/Oxi_Shapes_R4.py)) before the application was generated to automated the process and enable dynamic paramater tuning with an automatic shape download function at 300 DPI. This figure was constructed from these outputs and assembled in adobe illustrator. 
3. Static and dynamic geometry calculator. The [C_shapes](https://github.com/JamesCobley/Oxi_Shapes/blob/main/C_shapes.py) script prints the i-space cardinality, k-space strucutre (binomial coefficients), and allowed and barred transition as an excel file for any R interger. This was extended to an [app](https://cshapes.streamlit.app/).
4. Application. The Oxi-Shapes [Streamlit](https://streamlit.io/) app was generated using the [app](https://github.com/JamesCobley/Oxi_Shapes/blob/main/App.py) script. The application can be accessed for free [online](https://oxishapes.streamlit.app/). The application was used to create Extended Figure 3 and 4.
5. Simulations. 
# Resources 
This repository contains the white paper, where the detailed mathematical basis of Oxi-Shapes is elaborated (supplemental information), and the  methods section, which provides further details on the scripts above and general procedures. 
# Data Availability
This repository contains the published data, and their metadata, used to compute the Oxi-shapes mentioned in the preprint. Each of data file has been uploaded.
# Further Reading
For further reading one may wish to read the published papers on [oxiforms](https://onlinelibrary.wiley.com/doi/full/10.1002/bies.202200248), [the unmapped cysteine proteoform space](https://journals.physiology.org/doi/abs/10.1152/ajpcell.00152.2024), and the [nonlinear dynamics of cysteine proteoforms](https://www.sciencedirect.com/science/article/pii/S2213231725000369).
# Contact
For further information contact James Cobley (jcobley001@dundee.ac.uk) or (j_cobley@yahoo.com)
