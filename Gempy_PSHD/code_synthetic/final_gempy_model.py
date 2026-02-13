import numpy as np
import matplotlib.pyplot as plt

import gempy as gp
import gempy_viewer as gpv


def create_final_gempy_model_KSL_3_layer(refinement,filename,sp_cord, save=True):
    """ Create an initial gempy model objet

    Args:
        refinement (int): Refinement of grid
        sp_cord : coordinates dataset
        save (bool, optional): Whether you want to save the image

    """
    geo_model_test_post = gp.create_geomodel(
    project_name='Gempy_abc_Test_post',
    extent=[0, 1000, -10, 10, -900, -700],
    resolution=[100,10,100],
    refinement=7,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )
    
    brk1 = -845 
    brk2 = -825 
    

    gp.add_surface_points(
        geo_model=geo_model_test_post,
        x=[100.0,300, 900.0],
        y=[0.0,0.0, 0.0],
        z=[brk1,sp_cord[4,2], brk1],
        elements_names=['surface1','surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test_post,
        x=[800],
        y=[0.0],
        z=[brk1],
        elements_names=['surface1'],
        pole_vector=[[0, 0, 0.5]]
    )
    geo_model_test_post.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([100.0,300, 900.0]),
            y=np.array([0.0,0.0, 0.0]),
            z=np.array([brk2, sp_cord[1,2], brk2]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element2)

    num_elements = len(geo_model_test_post.structural_frame.structural_groups[0].elements) - 1  # Number of elements - 1 for zero-based index
    for swap_length in range(num_elements, 0, -1):  
        for i in range(swap_length):
            # Perform the swap for each pair (i, i+1)
            geo_model_test_post.structural_frame.structural_groups[0].elements[i], geo_model_test_post.structural_frame.structural_groups[0].elements[i + 1] = \
            geo_model_test_post.structural_frame.structural_groups[0].elements[i + 1], geo_model_test_post.structural_frame.structural_groups[0].elements[i]

    
    return geo_model_test_post
