import os
import sys
import json
import glob
import argparse
import traceback

# MFLM expects its own directory on PYTHONPATH.
sys.path.insert(0, os.path.abspath("MFLM"))
sys.path.insert(0, os.path.abspath("phase3"))  # prefer isolated Phase 3 cli_demo.py

import cli_demo


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dte-fdm-dir",
        default="./phase3/results/dte_fdm",
    )

    parser.add_argument(
        "--mflm-output",
        default="./phase3/results/refined_masks",
    )

    parser.add_argument(
        "--version",
        default="./weight/fakeshield-v1-22b/MFLM",
    )

    parser.add_argument(
        "--precision",
        default="bf16",
    )

    parser.add_argument(
        "--image_size",
        default=1024,
        type=int,
    )

    parser.add_argument(
        "--model_max_length",
        default=1536,
        type=int,
    )

    parser.add_argument(
        "--vision-tower",
        default="./clip-vit-large-patch14-336",
    )

    parser.add_argument(
        "--local-rank",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        choices=["llava_v1", "llava_llama_2"],
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Find DTE-FDM outputs
    # ---------------------------------------------------------

    dte_files = sorted(
        glob.glob(
            os.path.join(
                args.dte_fdm_dir,
                "*.json",
            )
        )
    )

    print("======== MFLM Batch Setup ========")
    print("DTE-FDM JSON files found:", len(dte_files))

    if len(dte_files) == 0:
        print("No DTE-FDM JSON files found.")
        return

    # ---------------------------------------------------------
    # Load DTE-FDM JSON files
    # ---------------------------------------------------------

    records = []

    for path in dte_files:
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        records.append(record)

    print("DTE-FDM records loaded:", len(records))

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    os.makedirs(
        args.mflm_output,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Prepare arguments expected by cli_demo.py
    # ---------------------------------------------------------

    mflm_args = argparse.Namespace(
        version=args.version,
        precision=args.precision,
        image_size=args.image_size,
        model_max_length=args.model_max_length,
        vision_tower=args.vision_tower,
        local_rank=args.local_rank,
        use_mm_start_end=True,
        conv_type=args.conv_type,
    )

    # ---------------------------------------------------------
    # LOAD MFLM MODEL ONCE
    # ---------------------------------------------------------

    print()
    print("======== MFLM Model Loading ========")

    tokenizer = cli_demo.setup_tokenizer_and_special_tokens(
        mflm_args
    )

    model = cli_demo.initialize_model(
        mflm_args,
        tokenizer,
    )

    model = cli_demo.prepare_model_for_inference(
        model,
        mflm_args,
    )

    global_enc_processor = cli_demo.CLIPImageProcessor.from_pretrained(
        model.config.vision_tower
    )

    transform = cli_demo.ResizeLongestSide(
        mflm_args.image_size
    )

    model.eval()

    print("======== MFLM Model Loaded ========")

    # ---------------------------------------------------------
    # Connect initialized objects to cli_demo's inference()
    # ---------------------------------------------------------
    #
    # cli_demo.inference() uses these as module globals.
    # We therefore put the already-loaded objects there.
    #

    cli_demo.args = mflm_args
    cli_demo.tokenizer = tokenizer
    cli_demo.model = model
    cli_demo.global_enc_processor = global_enc_processor
    cli_demo.transform = transform

    cli_demo.conv_history = {
        "user": [],
        "model": [],
    }

    cli_demo.mask_path = None

    # ---------------------------------------------------------
    # Process all DTE-FDM outputs
    # ---------------------------------------------------------

    print()
    print("======== MFLM Localization Begin ========")

    completed = 0
    skipped = 0
    failed = 0

    for index, record in enumerate(records, 1):
        input_image = record.get("image", "")
        input_text = record.get("outputs", "")

        filename = os.path.basename(input_image)

        output_path = os.path.join(
            args.mflm_output,
            filename,
        )

        # -----------------------------------------------------
        # Incremental behavior:
        # don't re-run an image whose mask already exists.
        # -----------------------------------------------------

        if os.path.exists(output_path):
            print(
                f"[{index}/{len(records)}] "
                f"SKIP {filename} "
                f"(already exists)"
            )

            skipped += 1
            continue

        # -----------------------------------------------------
        # DTE-FDM says the image is not manipulated.
        # No localization mask should be generated.
        # -----------------------------------------------------

        if "has not been tampered with" in input_text:
            print(
                f"[{index}/{len(records)}] "
                f"SKIP {filename} "
                f"(DTE-FDM: not tampered)"
            )

            skipped += 1
            continue

        print(
            f"[{index}/{len(records)}] "
            f"Processing {filename}"
        )

        try:

            # Reset conversation state for each independent image.
            cli_demo.conv_history = {
                "user": [],
                "model": [],
            }

            cli_demo.conv = None

            # -------------------------------------------------
            # EXACT EXISTING MFLM INFERENCE FUNCTION
            # -------------------------------------------------

            output_image, markdown_out = cli_demo.inference(
                input_text,
                {
                    "image": input_image,
                    "boxes": [],
                },
                False,
                False,
            )

            # -------------------------------------------------
            # Save predicted mask
            # -------------------------------------------------

            if output_image is None:
                print(
                    f"    WARNING: no mask generated for "
                    f"{filename}"
                )

                failed += 1
                continue

            output_image.save(
                output_path
            )

            print(
                f"    Mask saved: {output_path}"
            )

            completed += 1

        except Exception as e:

            failed += 1

            print(f"    ERROR processing {filename}: {type(e).__name__}: {e}")
            traceback.print_exc()

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("======== MFLM Batch Complete ========")
    print("Total DTE-FDM records:", len(records))
    print("Masks generated:", completed)
    print("Skipped:", skipped)
    print("Failed:", failed)
    print("Output directory:", args.mflm_output)


if __name__ == "__main__":
    main()
