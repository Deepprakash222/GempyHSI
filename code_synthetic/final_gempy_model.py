import numpy as np
import matplotlib.pyplot as plt

import gempy as gp
import gempy_viewer as gpv

def create_final_gempy_model_Salinas_6_layer(refinement,filename, sp_cord, save=True):
    """ Create an final gempy model objet

    Args:
        refinement (int): Refinement of grid
        sp_cord : coordinates dataset
        save (bool, optional): Whether you want to save the image

    """
    geo_model_test_post = gp.create_geomodel(
    project_name='Gempy_abc_Test_post',
    extent=[0, 86, -10, 10, -83, 0],
    resolution=[86,20,83],
    refinement=refinement,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )

    gp.add_surface_points(
        geo_model=geo_model_test_post,
        x=[70.0, 80.0],
        y=[0.0, 0.0],
        z=[-77.0, -71.0],
        elements_names=['surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test_post,
        x=[75],
        y=[0.0],
        z=[-74],
        elements_names=['surface1'],
        pole_vector=[[-5/3, 0, 1]]
    )
    geo_model_test_post.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([20.0, 60.0]),
            y=np.array([0.0, 0.0]),
            z=np.array([sp_cord[12,2], -52]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element2)

    element3 = gp.data.StructuralElement(
        name='surface3',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 30.0, 60]),
            y=np.array([0.0, 0.0,0.0]),
            z=np.array([-72, -55.5, -39]),
            names='surface3'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element3)

    element4 = gp.data.StructuralElement(
        name='surface4',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0, 60]),
            y=np.array([0.0, 0.0,0.0]),
            z=np.array([-61, sp_cord[7,2], -27]),
            names='surface4'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element4)

    element5 = gp.data.StructuralElement(
        name='surface5',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20, 40]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([-39, sp_cord[4,2], -16]),
            names='surface5'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element5)

    element6 = gp.data.StructuralElement(
        name='surface6',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0,30]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([-21, sp_cord[1,2], -1]),
            names='surface6'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element6)

    num_elements = len(geo_model_test_post.structural_frame.structural_groups[0].elements) - 1  # Number of elements - 1 for zero-based index
    for swap_length in range(num_elements, 0, -1):  
        for i in range(swap_length):
            # Perform the swap for each pair (i, i+1)
            geo_model_test_post.structural_frame.structural_groups[0].elements[i], geo_model_test_post.structural_frame.structural_groups[0].elements[i + 1] = \
            geo_model_test_post.structural_frame.structural_groups[0].elements[i + 1], geo_model_test_post.structural_frame.structural_groups[0].elements[i]

   
    return geo_model_test_post

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

def create_final_gempy_model_KSL_4_layer(refinement,filename, sp_cord, save=True):
    """ Create an initial gempy model objet

    Args:
        refinement (int): Refinement of grid
        save (bool, optional): Whether you want to save the image

    """
    geo_model_test_post = gp.create_geomodel(
    project_name='Gempy_abc_Test_post',
    extent=[0, 1000, -10, 10, -900, -700],
    resolution=[100,10,100],
    refinement=refinement,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )
    
    # brk1 = -855
    # brk2 = -845 
    # brk3 = -825 
    brk1 = -847
    brk2 = -824
    brk3 = -793

    gp.add_surface_points(
        geo_model=geo_model_test_post,
        x=[100.0,300, 900.0],
        y=[0.0,0.0, 0.0],
        z=[brk1,sp_cord[7,2], brk1],
        elements_names=['surface1','surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test_post,
        x=[800],
        y=[0.0],
        z=[brk1],
        elements_names=['surface1'],
        pole_vector=[[0, 0, 1]]
    )
    
    geo_model_test_post.update_transform(gp.data.GlobalAnisotropy.NONE)

    
    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([100.0,300, 900.0]),
            y=np.array([0.0,0.0, 0.0]),
            z=np.array([brk2, sp_cord[4,2], brk2]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element2)

    element3 = gp.data.StructuralElement(
        name='surface3',
        color=next(geo_model_test_post.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([100.0,300, 900.0]),
            y=np.array([0.0,0.0, 0.0]),
            z=np.array([brk3, sp_cord[1,2], brk3]),
            names='surface3'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test_post.structural_frame.structural_groups[0].append_element(element3)
    
    num_elements = len(geo_model_test_post.structural_frame.structural_groups[0].elements) - 1  # Number of elements - 1 for zero-based index
    
    for swap_length in range(num_elements, 0, -1):  
        for i in range(swap_length):
            # Perform the swap for each pair (i, i+1)
            geo_model_test_post.structural_frame.structural_groups[0].elements[i], geo_model_test_post.structural_frame.structural_groups[0].elements[i + 1] = \
            geo_model_test_post.structural_frame.structural_groups[0].elements[i + 1], geo_model_test_post.structural_frame.structural_groups[0].elements[i]
    
    return geo_model_test_post
    