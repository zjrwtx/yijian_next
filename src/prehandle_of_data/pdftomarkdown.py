from markitdown import MarkItDown

md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
result = md.convert("output.pdf")
print(result.text_content)

# markitdown path-to-file.pdf -o document.md
# from markitdown import MarkItDown
# from openai import OpenAI

# client = OpenAI()
# md = MarkItDown(llm_client=client, llm_model="gpt-4o")
# result = md.convert("test.pdf")
# print(result.text_content)


# pdfimagesocr 
# https://github.com/ocrmypdf/OCRmyPDF