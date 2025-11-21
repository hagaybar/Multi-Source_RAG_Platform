# Document Chunking Rules

This document defines the chunking strategies for different document types in the RAG system.

## Chunking Rules by Document Type

| Type | Split Strategy | Min Chunk Size | Notes |
|------|---------------|----------------|-------|
| email | split_on_blank_lines | "500" | ignore quoted replies, exclude headers/footers |
| docx | split_on_headings | "700" | collapse tables row-wise, include headers |
| pdf | split_on_pages | "800" | preserve page boundaries, include headers/footers |
| txt | split_on_blank_lines | "400" | simple text processing, exclude headers/footers |
| html | split_on_headings | "600" | preserve semantic structure, include headers |
| md | split_on_headings | "500" | markdown-aware splitting, include headers |
| csv | split_on_rows | "1000" | group related rows, detect delimiter automatically, include headers |
| json | split_on_objects | "300" | preserve object integrity, exclude headers/footers |
| xml | split_on_elements | "600" | maintain element hierarchy, include headers |
| pptx | split_on_slides | "400" | one chunk per slide, include headers |
| xlsx | split_on_sheets | "1200" | separate sheets, preserve structure, include headers |
| rtf | split_on_headings | "600" | similar to docx handling, include headers |
| log | split_on_timestamp | "500" | group by time periods, exclude headers/footers |
| py | split_on_functions | "300" | preserve function boundaries, exclude headers/footers |

## Split Strategy Definitions

### split_on_blank_lines
- Splits document at double line breaks or blank lines
- Preserves paragraph structure
- Good for plain text and email content

### split_on_headings
- Splits at heading markers (H1, H2, etc.)
- Maintains document hierarchy
- Ideal for structured documents

### split_on_pages
- Splits at page boundaries
- Preserves original pagination
- Important for PDFs with page-specific content

### split_on_rows
- Splits tabular data into logical groups
- Maintains column relationships
- Used for CSV and similar formats

### split_on_objects
- Splits at object boundaries in structured data
- Preserves data integrity
- Applied to JSON and similar formats

### split_on_elements
- Splits at major element boundaries
- Maintains XML/HTML structure
- Preserves semantic relationships

### split_on_slides
- Creates one chunk per presentation slide
- Maintains slide-level context
- Specific to presentation formats

### split_on_sheets
- Separates different worksheets
- Preserves sheet-level organization
- Used for spreadsheet formats

### split_on_timestamp
- Groups entries by time periods
- Maintains chronological context
- Ideal for log files

### split_on_functions
- Splits at function/method boundaries
- Preserves code structure
- Maintains functional context

## Header/Footer Inclusion Guidelines

Documents are processed with appropriate header/footer inclusion based on their type and structure needs. Headers are typically included for structured documents to preserve context, while footers are included when they contain important metadata or citations. Informal documents often exclude headers/footers to reduce noise and focus on main content.

## Special Processing Notes

- **Email**: Quoted reply sections are filtered out to avoid duplicate content
- **DOCX**: Tables are collapsed row-wise to maintain readability in chunks
- **PDF**: Page boundaries are preserved to maintain original document structure
- **CSV**: Related rows are grouped together to maintain data relationships, delimiter detection is automatic
- **JSON**: Object integrity is preserved to maintain data structure
- **Log**: Timestamp-based grouping maintains chronological context
- **Python Code**: Function boundaries are preserved to maintain code logic

## Chunk Size Considerations

- Minimum chunk sizes are set to ensure sufficient context
- Larger chunks (800-1200) for complex structured documents
- Smaller chunks (300-500) for simple or code-based content
- Balance between context preservation and processing efficiency