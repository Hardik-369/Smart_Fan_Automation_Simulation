def get_fan_speed(person_count: int) -> str:
    """
    Determine fan speed based on the number of detected persons.
    
    Args:
        person_count (int): Number of persons detected.
    
    Returns:
        str: Fan speed ('Off', 'Low', 'Medium', 'High').
    """
    if person_count == 0:
        return "Off"
    elif person_count == 1:
        return "Low"
    elif person_count == 2:
        return "Medium"
    else:
        return "High"
