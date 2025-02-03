import os
import shutil
import nibabel as nb
import numpy as np
import pandas as pd
import ants
import subprocess
from nipype.interfaces.fsl import RobustFOV
from nipype.interfaces.base import CommandLine
from nipype.interfaces.ants import Registration
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from filelock import FileLock

def apply_robustfov(input_file, output_file):
    """
    Apply FSL's robustfov to the input file.
    """
    robustfov = RobustFOV()
    robustfov.inputs.in_file = input_file
    robustfov.inputs.out_roi = output_file
    robustfov.inputs.out_transform = output_file
    robustfov.run()
    print(f"RobustFOV applied to {input_file}, output saved to {output_file}")
    return output_file

def apply_bias_correction(input_file, output_file):
    """
    Apply FreeSurfer's mri_nu_correct.mni to the input file and remove the log file after processing.
    """
    bias_correct = CommandLine(
        "mri_nu_correct.mni",
        args=f"--i {input_file} --o {output_file}",
    )
    bias_correct.run()
    print(f"Bias correction completed, output saved to {output_file}")

    log_file_path = os.path.join(os.path.dirname(output_file), "mri_nu_correct.mni.log")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    return output_file

def ants_linear_T1w_NM_registration(T1_corrected_path, NM_corrected_path, output_dir, subject_id):
    """
    Perform ANTs rigid registration to align the NM image to the T1-weighted image using the Nipype ANTs wrapper.
    """
    # Define output file paths
    output_prefix = os.path.join(output_dir, f"{subject_id}_NM_space-T1w")
    warped_image_path = f"{output_prefix}_Warped.nii.gz"
    inverse_warped_image_path = f"{output_prefix}_InverseWarped.nii.gz"
    affine_transform_path = f"{output_prefix}_Affine.mat"

    # Create registration object
    reg = Registration()
    reg.inputs.verbose = True
    reg.inputs.float = False

    # Set input images
    reg.inputs.fixed_image = T1_corrected_path
    reg.inputs.moving_image = NM_corrected_path

    # Set output transform prefix
    reg.inputs.output_transform_prefix = output_prefix

    # Define transformations
    reg.inputs.transforms = ["Rigid"]
    reg.inputs.transform_parameters = [(0.1,)]  # Equivalent to --transform Rigid[0.1]

    # Convergence parameters
    reg.inputs.number_of_iterations = [[100, 70, 50, 20]]  # Corresponds to --convergence [100x70x50x20, 1e-6, 10]
    reg.inputs.convergence_threshold = [1e-6]
    reg.inputs.convergence_window_size = [10]

    # Shrink factors and smoothing sigmas
    reg.inputs.shrink_factors = [[8, 4, 2, 1]]  # Corresponds to --shrink-factors 8x4x2x1
    reg.inputs.smoothing_sigmas = [[3, 2, 1, 0]]  # Corresponds to --smoothing-sigmas 3x2x1x0vox
    reg.inputs.sigma_units = ["vox"]

    # Similarity metric
    reg.inputs.metric = ["CC"]  # Corresponds to --metric CC[T1_corrected_path, NM_corrected_path,1,4]
    reg.inputs.metric_weight = [1]
    reg.inputs.radius_or_number_of_bins = [4]

    # Histogram matching & intensity settings
    reg.inputs.use_histogram_matching = [False]  # Corresponds to --use-histogram-matching 0
    reg.inputs.winsorize_lower_quantile = 0.005  # Corresponds to --winsorize-image-intensities [0.005, 0.995]
    reg.inputs.winsorize_upper_quantile = 0.995

    # Interpolation method
    reg.inputs.interpolation = "Linear"  # Corresponds to --interpolation Linear

    # Output file settings
    reg.inputs.output_warped_image = warped_image_path
    reg.inputs.output_inverse_warped_image = inverse_warped_image_path

    # Run the registration
    print(f"Beginning NM-to-T1 registration")
    print(reg.cmdline)
    reg.run()
    print(f"NM-to-T1 registration saved to {warped_image_path}")

def ants_normalization_to_MNI(T1_corrected_path, MNI_template_path, output_dir, subject_id, num_threads):
    """
    Perform ANTs SyN-based normalization to align the T1-weighted image to the MNI template.

    Parameters:
        T1_corrected_path (str): Path to the bias-corrected T1-weighted image.
        MNI_template_path (str): Path to the MNI template.
        output_dir (str): Directory where output files will be saved.
        subject_id (str): Subject ID to name the output files appropriately.
        num_threads (int): Number of threads to use for parallel processing.
    """
    # Define output file paths
    output_prefix = os.path.join(output_dir, f"{subject_id}_T1w_space-MNI152NLin2009cSym")
    warped_image_path = f"{output_prefix}_Warped.nii.gz"
    inverse_warped_image_path = f"{output_prefix}_InverseWarped.nii.gz"
    affine_transform_path = f"{output_prefix}_Affine.mat"
    warp_transform_path = f"{output_prefix}_Warp.nii.gz"
    inverse_warp_transform_path = f"{output_prefix}_InverseWarp.nii.gz"

    # Set number of threads for ANTs processing
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)

    # Print thread info
    print(f"Using {num_threads} threads for ANTs processing.")

    # Create a Registration object
    reg = Registration()
    reg.inputs.verbose = True  # Enable verbose output
    reg.inputs.num_threads = num_threads  # Set number of threads
    reg.inputs.dimension = 3  # Set image dimensionality
    reg.inputs.float = False

    # Set fixed (MNI template) and moving (T1-weighted) images
    reg.inputs.fixed_image = MNI_template_path
    reg.inputs.moving_image = T1_corrected_path

    # Output transform prefix
    reg.inputs.output_transform_prefix = output_prefix
    reg.inputs.output_warped_image = warped_image_path
    reg.inputs.output_inverse_warped_image = inverse_warped_image_path

    # Define transformations
    reg.inputs.transforms = ["Rigid", "Affine", "SyN"]
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3, 0)]
    reg.inputs.initial_moving_transform_com = 1
    reg.inputs.initialize_transforms_per_stage = False

    # Convergence parameters
    reg.inputs.number_of_iterations = [[1000, 500, 250, 100], [1000, 500, 250, 100], [100, 70, 50, 20]]
    reg.inputs.convergence_threshold = [1e-6, 1e-6, 1e-6]
    reg.inputs.convergence_window_size = [10, 10, 10]

    # Shrink factors & smoothing sigmas
    reg.inputs.shrink_factors = [[8, 4, 2, 1], [8, 4, 2, 1], [8, 4, 2, 1]]
    reg.inputs.smoothing_sigmas = [[3, 2, 1, 0], [3, 2, 1, 0], [3, 2, 1, 0]]
    reg.inputs.sigma_units = ["vox", "vox", "vox"]

    # Similarity metric
    reg.inputs.metric = ["MI", "MI", "CC"]
    reg.inputs.metric_weight = [1, 1, 1]
    reg.inputs.radius_or_number_of_bins = [32, 32, 4]
    reg.inputs.sampling_strategy = ["Regular", "Regular", None]
    reg.inputs.sampling_percentage = [0.25, 0.25, None]

    # Histogram matching & intensity settings
    reg.inputs.use_histogram_matching = [False, False, False]
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995

    # Interpolation method
    reg.inputs.interpolation = "Linear"

    # Run the registration
    print(f"Beginning T1-to-MNI normalization for subject {subject_id}")
    print(reg.cmdline)
    reg.run()
    print(f"T1-to-MNI normalization complete. Warped image saved at {warped_image_path}")

def transform_images_to_MNI_and_NM_space(
    midbrain_MNI_space_path,
    MNI_template_path,
    NM_input_path,
    warp_T1_to_MNI_path,
    inv_warp_T1_to_MNI_path,
    affine_T1_to_MNI_path,
    transform_NM_to_T1_path,
    output_dir,
    subject_id,
    interpolator
):
    """
    Transform images between MNI and NM space.
    """
    transformed_image = ants.apply_transforms(
        fixed=ants.image_read(NM_input_path),
        moving=ants.image_read(midbrain_MNI_space_path),
        transformlist=[
            transform_NM_to_T1_path,
            affine_T1_to_MNI_path,
            inv_warp_T1_to_MNI_path
        ],
        whichtoinvert=[True, True, False],
        interpolator=interpolator,
    )
    
    output_path = os.path.join(output_dir, f"{subject_id}_midbrain_atlas_space-NM.nii.gz")
    transformed_image.to_filename(output_path)
    print(f"Midbrain atlas transformed into NM-MRI space using {interpolator} saved at {output_path}")
    

def compute_CNR_map(
    NM_input_path, midbrain_NM_space_path, NM_CNR_path, NM_CNR_MNI_space_path,
    MNI_template_path, warp_T1_to_MNI_path, affine_T1_to_MNI_path,
    transform_NM_to_T1_path, subject_id
):
    """
    Compute and save the Contrast-to-Noise Ratio (CNR) map for NM-MRI and move it to MNI space.
    """
    if not os.path.isfile(NM_input_path) or not os.path.isfile(midbrain_NM_space_path):
        print(f"Missing input files: {NM_input_path}, {midbrain_NM_space_path}")
        return

    NM_nii = nb.load(NM_input_path)
    midbrain_nii = nb.load(midbrain_NM_space_path).get_fdata()
    NM_data = NM_nii.get_fdata()
    
    CC_mask = (midbrain_nii == 1)
    if not CC_mask.any():
        print(f"Warning: No mask values found in CC mask for subject {subject_id}")
        return

    CC_mode = refined_mode_estimation(NM_data[CC_mask])
    NM_CNR = (NM_data - CC_mode) / CC_mode
    nb.save(nb.Nifti1Image(NM_CNR, affine=NM_nii.affine), NM_CNR_path)
    print(f"CNR map saved to {NM_CNR_path}")

    # Move NM-CNR image to MNI space
    transformed_image = ants.apply_transforms(
        fixed=ants.image_read(MNI_template_path),
        moving=ants.image_read(NM_CNR_path),
        transformlist=[
            warp_T1_to_MNI_path,
            affine_T1_to_MNI_path,
            transform_NM_to_T1_path
        ],
        whichtoinvert=[False, False, False],
        interpolator="linear"
    )

    transformed_image.to_filename(NM_CNR_MNI_space_path)
    print(f"NM-CNR image transformed to MNI space saved at {NM_CNR_MNI_space_path}")

def refined_mode_estimation(array, cut_down=True, bw_method="scott"):
    """
    Estimate the mode of an array using refined KDE-based optimization.
    """
    kernel = kde(array, cut_down=cut_down, bw_method=bw_method)
    x0 = array[np.argmax(kernel.pdf(array))]
    results = minimize(lambda x: -kernel(x)[0], x0=x0, bounds=[[array.min(), array.max()]])
    return results.x[0]

def kde(array, cut_down=True, bw_method="scott"):
    """
    Compute the kernel density estimation (KDE) for a given array.
    """
    if cut_down:
        bins, counts = np.unique(array, return_counts=True)
        threshold = bins[counts > counts.mean()]
        array = array[(threshold.min() < array) & (array < threshold.max())]
    return gaussian_kde(array, bw_method=bw_method)

def save_SNVTA_results(NM_CNR_path, midbrain_NM_space_path, results_dir, subject_id):
    """
    Save mean and standard deviation of values in SNVTA mask using a file lock.
    This prevents race conditions when multiple processes are writing to the same file.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "NM_CNR_SNVTA_Results.xlsx")
    lock_file = results_file + ".lock"  # Lock file to prevent corruption

    NM_CNR_img = nb.load(NM_CNR_path).get_fdata()
    midbrain_img = nb.load(midbrain_NM_space_path).get_fdata()
    mask_values = NM_CNR_img[midbrain_img == 2]

    mean_value = np.mean(mask_values)
    std_value = np.std(mask_values)

    # Use FileLock to ensure only one process writes to the file at a time
    with FileLock(lock_file):
        results_df = pd.read_excel(results_file) if os.path.exists(results_file) else pd.DataFrame(columns=["Subject ID", "Mean", "Standard Deviation"])
        
        if subject_id in results_df["Subject ID"].values:
            results_df.loc[results_df["Subject ID"] == subject_id, ["Mean", "Standard Deviation"]] = [mean_value, std_value]
        else:
            results_df = pd.concat([results_df, pd.DataFrame([{"Subject ID": subject_id, "Mean": mean_value, "Standard Deviation": std_value}])], ignore_index=True)

        results_df.to_excel(results_file, index=False)
    
    print(f"Results saved to {results_file} (processed subject {subject_id})")
