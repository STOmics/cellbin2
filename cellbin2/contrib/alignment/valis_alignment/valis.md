## VALIS Image Alignment Tool

Tool for calculating pixel offsets between two images using the VALIS registration library.

## Tutorial

### 1. Environment Setup

- Download appropriate JDK from [java downloads](https://www.oracle.com/java/technologies/downloads/)
- Edit your system and environment variables to update the Java home
    ```shell
    # Windows:
    setx JAVA_HOME "/path/to/jdk"

    # Linux/Mac:
    export JAVA_HOME=/path/to/jdk
    ```

- Verify the path has been added
    ```shell
    # Windows:
    echo %JAVA_HOME%

    # Linux/Mac:
    echo $JAVA_HOME
    ```

- Python 3.9 or 3.10

    ```shell
    pip install valis-wsi
    ```
### 2. Usage
By inputting the reference image and moving image, this tool will return the offset, both x and y direction, of the moving image.
### Command Line Arguments

| Argument | Short | Description | Required |
|:---|:---|:---|:---|
| `--reference` | `-r` | Path to reference image | ✓ |
| `--move` | `-m` | Path to moving image | ✓ |

### Basic Command
```bash
python valis_alignment.py -r <reference_image> -m <moving_image>
```

### Example Output

```bash
X offset:    -15.23 pixels
Y offset:    234.56 pixels
```

## Notes 

- Supports TIFF, JPEG, PNG, and other common image formats

- Ensure both images have sufficient features for accurate registration