if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found: {model_path}"
    )