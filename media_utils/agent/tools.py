"""LangChain tools for media generation and OCR."""

from pathlib import Path
from typing import Optional, Literal
from datetime import datetime

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ImageGenerationInput(BaseModel):
    """Input schema for image generation tool."""

    prompt: str = Field(description="Text description of the image to generate")
    negative_prompt: Optional[str] = Field(
        default=None,
        description="What to avoid in the image (e.g., 'blurry, low quality')"
    )
    resolution: str = Field(
        default="1024x1024",
        description="Image resolution. Options: '1024x1024', '1344x768' (landscape), '768x1344' (portrait)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )


class OCRInput(BaseModel):
    """Input schema for OCR tool."""

    image_path: str = Field(description="Path to the image file to extract text from")
    mode: Literal["markdown", "free", "figure", "describe"] = Field(
        default="markdown",
        description="OCR mode: 'markdown' (structured), 'free' (plain text), 'figure' (charts), 'describe' (image description)"
    )


class ImageGenerationTool(BaseTool):
    """Tool for generating images from text prompts."""

    name: str = "generate_image"
    description: str = """Generate an image from a text description.
    Use this when you need to create, generate, or produce an image based on a prompt.
    Returns the path to the generated image file."""
    args_schema: type[BaseModel] = ImageGenerationInput

    _generator: Optional[object] = None
    _output_dir: Path = Path("output")

    def __init__(self, output_dir: str = "output", **kwargs):
        super().__init__(**kwargs)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _get_generator(self):
        """Lazy-load the image generator."""
        if self._generator is None:
            from media_utils.image.generator import ImageGenerator
            self._generator = ImageGenerator(
                mode="pipeline",
                offload_mode="model",
                keep_loaded=True,
            )
        return self._generator

    def _run(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        resolution: str = "1024x1024",
        seed: Optional[int] = None,
    ) -> str:
        """Generate an image and return the file path."""
        generator = self._get_generator()

        # Generate image
        image = generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            seed=seed,
        )

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        output_path = self._output_dir / filename
        image.save(output_path)

        return f"Image generated and saved to: {output_path}"

    def unload(self):
        """Unload the generator to free GPU memory."""
        if self._generator is not None:
            self._generator.unload()
            self._generator = None


class OCRTool(BaseTool):
    """Tool for extracting text from images using OCR."""

    name: str = "extract_text"
    description: str = """Extract text from an image using OCR.
    Use this when you need to read, extract, or recognize text from an image or document.
    Returns the extracted text content."""
    args_schema: type[BaseModel] = OCRInput

    _ocr: Optional[object] = None

    def __init__(self, quantization: str = "4bit", **kwargs):
        super().__init__(**kwargs)
        self._quantization = quantization

    def _get_ocr(self):
        """Lazy-load the OCR model."""
        if self._ocr is None:
            from media_utils.ocr.deepseek import DeepSeekOCR
            self._ocr = DeepSeekOCR(
                quantization=self._quantization,
                keep_loaded=True,
            )
        return self._ocr

    def _run(
        self,
        image_path: str,
        mode: str = "markdown",
    ) -> str:
        """Extract text from an image."""
        ocr = self._get_ocr()

        # Verify file exists
        if not Path(image_path).exists():
            return f"Error: Image file not found: {image_path}"

        # Extract text
        text = ocr.extract(
            image_path,
            mode=mode,
            resolution="gundam",
        )

        return text

    def unload(self):
        """Unload the OCR model to free GPU memory."""
        if self._ocr is not None:
            self._ocr.unload()
            self._ocr = None


def create_tools(
    output_dir: str = "output",
    ocr_quantization: str = "4bit",
) -> list[BaseTool]:
    """Create all media tools.

    Args:
        output_dir: Directory for generated images.
        ocr_quantization: Quantization for OCR model ("4bit", "8bit", or None).

    Returns:
        List of LangChain tools.
    """
    return [
        ImageGenerationTool(output_dir=output_dir),
        OCRTool(quantization=ocr_quantization),
    ]
