"""Tests for the MediaAgent and tools."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from media_utils.agent.tools import (
    ImageGenerationTool,
    OCRTool,
    create_tools,
    ImageGenerationInput,
    OCRInput,
)
from media_utils.agent.graph import MediaAgent


class TestImageGenerationTool:
    """Tests for ImageGenerationTool."""

    def test_tool_name_and_description(self):
        """Test tool has correct name and description."""
        tool = ImageGenerationTool()
        assert tool.name == "generate_image"
        assert "generate" in tool.description.lower()
        assert "image" in tool.description.lower()

    def test_tool_args_schema(self):
        """Test tool has correct args schema."""
        tool = ImageGenerationTool()
        assert tool.args_schema == ImageGenerationInput

    def test_run_generates_image(self, temp_dir):
        """Test that tool generates and saves an image."""
        tool = ImageGenerationTool(output_dir=str(temp_dir))

        # Mock the generator
        mock_image = MagicMock()
        mock_generator = MagicMock()
        mock_generator.generate.return_value = mock_image

        with patch.object(tool, '_get_generator', return_value=mock_generator):
            result = tool._run(prompt="A test image")

            mock_generator.generate.assert_called_once()
            mock_image.save.assert_called_once()
            assert "generated_" in result
            assert ".png" in result

    def test_unload_clears_generator(self):
        """Test that unload clears the generator."""
        tool = ImageGenerationTool()
        mock_generator = MagicMock()
        tool._generator = mock_generator

        tool.unload()

        mock_generator.unload.assert_called_once()
        assert tool._generator is None


class TestOCRTool:
    """Tests for OCRTool."""

    def test_tool_name_and_description(self):
        """Test tool has correct name and description."""
        tool = OCRTool()
        assert tool.name == "extract_text"
        assert "extract" in tool.description.lower()
        assert "text" in tool.description.lower()

    def test_tool_args_schema(self):
        """Test tool has correct args schema."""
        tool = OCRTool()
        assert tool.args_schema == OCRInput

    def test_run_extracts_text(self, temp_dir):
        """Test that tool extracts text from image."""
        tool = OCRTool()

        # Create a fake image file
        image_path = temp_dir / "test.png"
        image_path.touch()

        # Mock the OCR
        mock_ocr = MagicMock()
        mock_ocr.extract.return_value = "Extracted text content"

        with patch.object(tool, '_get_ocr', return_value=mock_ocr):
            result = tool._run(image_path=str(image_path))

            mock_ocr.extract.assert_called_once()
            assert result == "Extracted text content"

    def test_run_returns_error_for_missing_file(self):
        """Test that tool returns error for missing file."""
        tool = OCRTool()

        result = tool._run(image_path="/nonexistent/file.png")

        assert "Error" in result
        assert "not found" in result

    def test_unload_clears_ocr(self):
        """Test that unload clears the OCR model."""
        tool = OCRTool()
        mock_ocr = MagicMock()
        tool._ocr = mock_ocr

        tool.unload()

        mock_ocr.unload.assert_called_once()
        assert tool._ocr is None


class TestCreateTools:
    """Tests for create_tools function."""

    def test_creates_all_tools(self):
        """Test that create_tools returns all tools."""
        tools = create_tools()

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "generate_image" in tool_names
        assert "extract_text" in tool_names

    def test_custom_output_dir(self, temp_dir):
        """Test that custom output dir is passed to image tool."""
        tools = create_tools(output_dir=str(temp_dir))

        image_tool = next(t for t in tools if t.name == "generate_image")
        assert image_tool._output_dir == temp_dir


class TestMediaAgent:
    """Tests for MediaAgent."""

    def test_agent_init(self):
        """Test agent initialization."""
        agent = MediaAgent(
            model_name="test/model",
            output_dir="test_output",
            ocr_quantization="4bit",
        )

        assert agent.model_name == "test/model"
        assert agent.output_dir == "test_output"
        assert len(agent.tools) == 2

    def test_agent_has_tools(self):
        """Test agent has correct tools."""
        agent = MediaAgent()

        assert "generate_image" in agent.tool_map
        assert "extract_text" in agent.tool_map

    def test_parse_tool_call_valid(self):
        """Test parsing a valid tool call."""
        agent = MediaAgent()

        response = """TOOL_CALL: generate_image
ARGUMENTS:
  prompt: A beautiful sunset
  resolution: 1024x1024"""

        result = agent._parse_tool_call(response)

        assert result is not None
        assert result["name"] == "generate_image"
        assert result["arguments"]["prompt"] == "A beautiful sunset"
        assert result["arguments"]["resolution"] == "1024x1024"

    def test_parse_tool_call_with_seed(self):
        """Test parsing tool call with integer argument."""
        agent = MediaAgent()

        response = """TOOL_CALL: generate_image
ARGUMENTS:
  prompt: Test image
  seed: 42"""

        result = agent._parse_tool_call(response)

        assert result is not None
        assert result["arguments"]["seed"] == 42  # Should be int

    def test_parse_tool_call_no_tool(self):
        """Test parsing response without tool call."""
        agent = MediaAgent()

        response = "This is just a normal response without any tool call."

        result = agent._parse_tool_call(response)

        assert result is None

    def test_parse_tool_call_ocr(self):
        """Test parsing OCR tool call."""
        agent = MediaAgent()

        response = """TOOL_CALL: extract_text
ARGUMENTS:
  image_path: document.png
  mode: markdown"""

        result = agent._parse_tool_call(response)

        assert result is not None
        assert result["name"] == "extract_text"
        assert result["arguments"]["image_path"] == "document.png"
        assert result["arguments"]["mode"] == "markdown"

    def test_get_system_prompt(self):
        """Test that system prompt contains tool information."""
        agent = MediaAgent()

        prompt = agent._get_system_prompt()

        assert "generate_image" in prompt
        assert "extract_text" in prompt
        assert "TOOL_CALL" in prompt

    def test_context_manager(self):
        """Test agent works as context manager."""
        with patch.object(MediaAgent, 'unload') as mock_unload:
            with MediaAgent() as agent:
                pass

            mock_unload.assert_called_once()

    def test_unload_clears_all(self):
        """Test that unload clears all models."""
        agent = MediaAgent()

        # Mock the tool internals
        for tool in agent.tools:
            if hasattr(tool, '_generator'):
                tool._generator = MagicMock()
            if hasattr(tool, '_ocr'):
                tool._ocr = MagicMock()

        agent._qwen = MagicMock()

        agent.unload()

        # Verify tools were cleared
        for tool in agent.tools:
            if hasattr(tool, '_generator'):
                assert tool._generator is None
            if hasattr(tool, '_ocr'):
                assert tool._ocr is None
