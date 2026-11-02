import torch


def q045(source, destination):
    """Transfer model state through an in-memory checkpoint. Destination must have the same
    parameter and buffer structure. Return the updated destination.
    """
    import io

    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    destination.load_state_dict(
        torch.load(buffer, map_location="cpu", weights_only=True)
    )
    return destination
