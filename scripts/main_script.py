import argparse
import os
from image_processing_functions import (
    apply_robustfov,
    apply_bias_correction,
    ants_linear_T1w_NM_registration,
    ants_normalization_to_MNI,
    transform_images_to_MNI_and_NM_space,
    compute_CNR_map,
    save_SNVTA_results
)

def process_subject(nifti_dir, output_dir, subject_id, templates_dir, results_dir, apply_robustfov_flag, num_threads, interpolator):
    """
    Process a single subject's images based on the input and output directories.
    """
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)  # Set number of threads for ANTs
    print("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:", os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"))
    
    subject_nifti_path = os.path.join(nifti_dir, subject_id, 'anat')
    subject_output_path = os.path.join(output_dir, subject_id, 'anat')

    # Define file paths
    T1_input_path = os.path.join(subject_nifti_path, f"{subject_id}_T1w.nii.gz")
    NM_input_path = os.path.join(subject_nifti_path, f"{subject_id}_NM.nii.gz")
    T1_fov_path = os.path.join(subject_output_path, f"{subject_id}_T1w_fov.nii.gz")
    T1_corrected_path = os.path.join(subject_output_path, f"{subject_id}_T1w_biascorr.nii.gz")
    NM_corrected_path = os.path.join(subject_output_path, f"{subject_id}_NM_biascorr.nii.gz")
    midbrain_MNI_space_path = os.path.join(templates_dir, "midbrain_atlas_space-MNI152NLin2009cSym.nii.gz")
    MNI_template_path = os.path.join(templates_dir, "mni_icbm152_t1_tal_nlin_sym_09c.nii")
    warp_T1_to_MNI_path = os.path.join(subject_output_path, f"{subject_id}_T1w_space-MNI152NLin2009cSym1Warp.nii.gz")
    inv_warp_T1_to_MNI_path = os.path.join(subject_output_path, f"{subject_id}_T1w_space-MNI152NLin2009cSym1InverseWarp.nii.gz")
    affine_T1_to_MNI_path = os.path.join(subject_output_path, f"{subject_id}_T1w_space-MNI152NLin2009cSym0GenericAffine.mat")
    transform_NM_to_T1_path = os.path.join(subject_output_path, f"{subject_id}_NM_space-T1w0GenericAffine.mat")
    midbrain_NM_space_path = os.path.join(subject_output_path, f"{subject_id}_midbrain_atlas_space-NM.nii.gz")
    NM_CNR_path = os.path.join(subject_output_path, f"{subject_id}_NM-CNR.nii.gz")
    NM_CNR_MNI_space_path = os.path.join(subject_output_path, f"{subject_id}_NM-CNR_space-MNI152NLin2009cSym.nii.gz")

    # Create output directories if they don't exist
    os.makedirs(subject_output_path, exist_ok=True)

    # Process T1-weighted file
    T1_processed_path = T1_input_path
    if apply_robustfov_flag == "True":
        T1_processed_path = apply_robustfov(T1_input_path, T1_fov_path)
    apply_bias_correction(T1_processed_path, T1_corrected_path)

    # Process NM file
    apply_bias_correction(NM_input_path, NM_corrected_path)

    # Perform registrations
    ants_linear_T1w_NM_registration(T1_corrected_path, NM_corrected_path, subject_output_path, subject_id)
    ants_normalization_to_MNI(T1_corrected_path, MNI_template_path, subject_output_path, subject_id, num_threads)

    # Transform midbrain atlas and NM images
    transform_images_to_MNI_and_NM_space(
        midbrain_MNI_space_path,
        MNI_template_path,
        NM_input_path,
        warp_T1_to_MNI_path,
        inv_warp_T1_to_MNI_path,
        affine_T1_to_MNI_path,
        transform_NM_to_T1_path,
        subject_output_path,
        subject_id,
        interpolator
    )

    # Compute and save CNR map
    compute_CNR_map(
        NM_input_path,
        midbrain_NM_space_path,
        NM_CNR_path,
        NM_CNR_MNI_space_path,
        MNI_template_path,
        warp_T1_to_MNI_path,
        affine_T1_to_MNI_path,
        transform_NM_to_T1_path,
        subject_id
    )

    # Calculate and save SNVTA results
    save_SNVTA_results(NM_CNR_path, midbrain_NM_space_path, results_dir, subject_id)

    print(f"Processing complete for subject: {subject_id}")

def process_all_subjects(nifti_dir, output_dir, subject_list_path, templates_dir, results_dir, apply_robustfov_flag, num_threads, interpolator):
    """
    Process all subjects specified in the subject list.
    """
    with open(subject_list_path, "r") as f:
        subject_ids = [line.strip() for line in f if line.strip()]

    for subject_id in subject_ids:
        print(f"Processing subject: {subject_id}")
        process_subject(nifti_dir, output_dir, subject_id, templates_dir, results_dir, apply_robustfov_flag, num_threads, interpolator)

def main():
    """
    Main function to handle user input and call processing functions.
    """
    parser = argparse.ArgumentParser(description="Process subjects with Nipype tools.")
    parser.add_argument("--nifti_dir", required=True, help="Path to the NIFTI directory.")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory.")
    parser.add_argument("--templates_dir", required=True, help="Path to the templates directory.")
    parser.add_argument("--results_dir", required=True, help="Path to the results directory.")
    parser.add_argument("--subject_list", required=True, help="Path to the file with the list of subjects to process.")
    parser.add_argument("--apply_robustfov", type=str, choices=["True", "False"], default="False",
                        help="Apply robustfov to T1 files. Default is False.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of CPU threads per subject.")
    parser.add_argument("--interpolator", type=str, default="genericLabel", help="Interpolation method (default: Linear).")
    args = parser.parse_args()
    
    apply_robustfov_flag = args.apply_robustfov == "True"

    process_all_subjects(
        args.nifti_dir,
        args.output_dir,
        args.subject_list,
        args.templates_dir,
        args.results_dir,
        args.apply_robustfov,
        args.num_threads,
        args.interpolator
    )

if __name__ == "__main__":
    main()
