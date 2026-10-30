import numpy as np
import matplotlib.pyplot as plt

import gempy as gp
import gempy_viewer as gpv
import torch
from gempy_engine.core.backend_tensor import BackendTensor

def create_gempy_model_KSL_initial(refinement,filename, depth_shift =None, save=True):
    """ Create an initial gempy model object.
 
    Args:
        refinement (int): Refinement of grid.
        filename (str): Where to save the plot, if `save=True`.
        depth_shift (sequence of 2 floats, optional): Offset applied to
            surface1's and surface2's middle interface point respectively.
            Defaults to (0, 0), i.e. no shift, if not provided.
        save (bool, optional): Whether you want to save the image.
    """
    if depth_shift is None:
        depth_shift = (0.0, 0.0)
        
    geo_model_test = gp.create_geomodel(
    project_name='Gempy_abc_Test',
    extent=[0, 1000, -10, 10, -900, -700],
    resolution=[100,10,100],
    refinement=refinement,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )
   
    brk1 = -845 
    brk2 = -825 
    
    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[100.0,300, 900.0],
        y=[0.0, 0.0, 0.0],
        z=[brk1, brk1+ depth_shift[0], brk1],
        elements_names=['surface1','surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[800],
        y=[0.0],
        z=[brk1],
        elements_names=['surface1'],
        pole_vector=[[0, 0, 0.5]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([100.0, 300.0,900.0]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([brk2, brk2+depth_shift[1], brk2]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element2)

    geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1] = \
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0]

    gp.compute_model(geo_model_test)
    picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    if save:
        plt.savefig(filename)
    
    return geo_model_test

def create_gempy_model_KSL_final(final_data, refinement=7,  filename='posterior_model.png', save=True):
    """ Create an initial gempy model objet

    Args:
        refinement (int): Refinement of grid
        save (bool, optional): Whether you want to save the image

    """
    geo_model_test = gp.create_geomodel(
    project_name='Gempy_abc_Test',
    extent=[0, 1000, -10, 10, -900, -700],
    resolution=[100,10,100],
    refinement=refinement,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )
   
    brk1 = -845 
    brk2 = -825 
    
    
    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[100.0,300, 900.0],
        y=[0.0, 0.0, 0.0],
        z=[final_data[3,2], final_data[4,2], final_data[5,2]],
        elements_names=['surface1','surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[800],
        y=[0.0],
        z=[brk1],
        elements_names=['surface1'],
        pole_vector=[[0, 0, 0.5]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([100.0, 300.0,900.0]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([final_data[0,2], final_data[1,2], final_data[2,2]]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element2)

    geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1] = \
    geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0]

    gp.compute_model(geo_model_test)
    picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    if save:
        print(filename)
        df_sp_final = geo_model_test.surface_points.df
        df_sp_final.to_csv(filename)  
        plt.savefig(filename)
    
    return geo_model_test


def create_gempy_model_Salinas_A_layer_initial(refinement,filename, save=True):
    """ Create an initial gempy model objet
    
        Args:
            refinement (int): Refinement of grid
            save (bool, optional): Whether you want to save the image
    
    """
    geo_model_test = gp.create_geomodel(
    project_name='Gempy_abc_Test',
    extent=[0, 86, -10, 10, -83, 0],
    resolution=[86,20,83],
    refinement=refinement,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )

    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[70.0, 80.0],
        y=[0.0, 0.0],
        z=[-77.0, -71.0],
        elements_names=['surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[75],
        y=[0.0],
        z=[-74],
        elements_names=['surface1'],
        pole_vector=[[-5/3, 0, 1]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([20.0, 60.0]),
            y=np.array([0.0, 0.0]),
            z=np.array([-74, -52]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element2)

    element3 = gp.data.StructuralElement(
        name='surface3',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 30.0, 60]),
            y=np.array([0.0, 0.0,0.0]),
            z=np.array([-72, -55.5, -39]),
            names='surface3'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element3)

    element4 = gp.data.StructuralElement(
        name='surface4',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0, 60]),
            y=np.array([0.0, 0.0,0.0]),
            z=np.array([-61, -49, -27]),
            names='surface4'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element4)

    element5 = gp.data.StructuralElement(
        name='surface5',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0, 40]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([-39, -28, -16]),
            names='surface5'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element5)

    element6 = gp.data.StructuralElement(
        name='surface6',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0,30]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([-21, -10, -1]),
            names='surface6'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element6)

    
    
    num_elements = len(geo_model_test.structural_frame.structural_groups[0].elements) - 1  # Number of elements - 1 for zero-based index
    
    for swap_length in range(num_elements, 0, -1):  
        for i in range(swap_length):
            # Perform the swap for each pair (i, i+1)
            geo_model_test.structural_frame.structural_groups[0].elements[i], geo_model_test.structural_frame.structural_groups[0].elements[i + 1] = \
            geo_model_test.structural_frame.structural_groups[0].elements[i + 1], geo_model_test.structural_frame.structural_groups[0].elements[i]



    gp.compute_model(geo_model_test)
    picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    if save:
        plt.savefig(filename)
        plt.close()
    # gpv.plot_3d(geo_model_test, cell_number=5, legend='force')
    return geo_model_test

def create_gempy_model_Salinas_A_layer_final(final_data, refinement=7,  filename='posterior_model.png', save=True):
    """ Create an initial gempy model objet

    Args:
        refinement (int): Refinement of grid
        save (bool, optional): Whether you want to save the image

    """
    geo_model_test = gp.create_geomodel(
    project_name='Gempy_abc_Test',
    extent=[0, 86, -10, 10, -83, 0],
    resolution=[86,20,83],
    refinement=refinement,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )

    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[70.0, 80.0],
        y=[0.0, 0.0],
        z=[final_data[14,2], final_data[15,2]],
        elements_names=['surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[75],
        y=[0.0],
        z=[-74],
        elements_names=['surface1'],
        pole_vector=[[-5/3, 0, 1]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([20.0, 60.0]),
            y=np.array([0.0, 0.0]),
            z=np.array([final_data[12,2], final_data[13,2]]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element2)

    element3 = gp.data.StructuralElement(
        name='surface3',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 30.0, 60]),
            y=np.array([0.0, 0.0,0.0]),
            z=np.array([final_data[9,2], final_data[10,2], final_data[11,2]]),
            names='surface3'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element3)

    element4 = gp.data.StructuralElement(
        name='surface4',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0, 60]),
            y=np.array([0.0, 0.0,0.0]),
            z=np.array([final_data[6,2], final_data[7,2],final_data[8,2]]),
            names='surface4'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element4)

    element5 = gp.data.StructuralElement(
        name='surface5',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0, 40]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([final_data[3,2], final_data[4,2],final_data[5,2]]),
            names='surface5'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element5)

    element6 = gp.data.StructuralElement(
        name='surface6',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([0.0, 20.0,30]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([final_data[0,2], final_data[1,2],final_data[2,2]]),
            names='surface6'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model_test.structural_frame.structural_groups[0].append_element(element6)

    
    
    num_elements = len(geo_model_test.structural_frame.structural_groups[0].elements) - 1  # Number of elements - 1 for zero-based index
    
    for swap_length in range(num_elements, 0, -1):  
        for i in range(swap_length):
            # Perform the swap for each pair (i, i+1)
            geo_model_test.structural_frame.structural_groups[0].elements[i], geo_model_test.structural_frame.structural_groups[0].elements[i + 1] = \
            geo_model_test.structural_frame.structural_groups[0].elements[i + 1], geo_model_test.structural_frame.structural_groups[0].elements[i]



    gp.compute_model(geo_model_test)
    picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    if save:
        df_sp_init = geo_model_test.surface_points.df
        df_sp_init.to_csv(filename)  
        plt.savefig(filename)
        plt.close()
    # gpv.plot_3d(geo_model_test, cell_number=5, legend='force')
    return geo_model_test




def generate_final_model(
    geo_model,              # already-built initial GemPy model
    geo_model_final_fn,     # callable: e.g. create_gempy_model_KSl_final,
                             # signature (final_data, refinement, filename, save) -> GeoModel
    random_variable,   # random_variable specs
    num_layers: int,
    posterior_data,
    slope: float = 200,
    filename: str = "posterior_model.png",
    save: bool = True,
):
    """
    Update `geo_model`'s surface points using `posterior_data`, then build
    the "final" model by calling the user-supplied `geo_model_final_fn`.

    This replaces the previous attempt to derive the final-model function
    from `geo_model`'s own name (`geo_model + "_final"`), which doesn't
    work -- GeoModel objects can't be concatenated with a string, and
    even if they could, there's no mechanism turning that string into a
    callable function. Passing the final-model builder in explicitly
    (as done everywhere else in this pipeline -- see `run_bayesian`,
    `build_and_save_realization`) is both correct and consistent with
    the rest of the design: it also means this function works unchanged
    for any dataset (KSL, Salinas, ...) -- just pass a different
    `geo_model_final_fn`.

    Args:
        geo_model: The already-built initial GemPy model whose surface
            points will be updated in place.
        geo_model_final_fn: Callable that builds the final model from
            physical-space interface coordinates. Expected signature:
            `geo_model_final_fn(final_data, refinement, filename, save) -> GeoModel`
            (matching `create_gempy_model_KSl_final` / its Salinas
            equivalent).
        random_variable: List of interface-update specs (same
            format as `random_variable`/`test_list` elsewhere), only the
            first `num_layers` entries are used.
        num_layers: Number of interface points being updated.
        posterior_data: Sequence of posterior Z-values (or X/Y, per each
            spec's "direction"), one per entry in `random_variable`.
        slope: sigmoid_slope to use when recomputing `geo_model`.
        filename: Where the final model's plot is saved.
        save: Whether to save the final model's plot.

    Returns:
        The rebuilt "final" GemPy model.
    """
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)

    interpolation_input = geo_model.interpolation_input
    geo_model.interpolation_options.sigmoid_slope = slope

    for index, interpolation_input_data in enumerate(random_variable[:num_layers]):
        direction = interpolation_input_data["direction"]
        if direction == "X":
            axis = 0
        elif direction == "Y":
            axis = 1
        elif direction == "Z":
            axis = 2
        else:
            raise ValueError(f"Unknown direction '{direction}' (expected 'X', 'Y', or 'Z').")

        interpolation_input.surface_points.sp_coords = torch.index_put(
            interpolation_input.surface_points.sp_coords,
            (torch.as_tensor(interpolation_input_data["id"]), torch.tensor([axis])),
            posterior_data[index],
        )

    physical_data = geo_model.transform.apply_inverse(
        interpolation_input.surface_points.sp_coords.detach().numpy()
    )
    geo_model_test = geo_model_final_fn(physical_data, refinement=7, save=save, filename=filename)
    return geo_model_test