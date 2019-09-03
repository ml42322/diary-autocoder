# Consumer Expenditure Diary Autocoder

Developed by Michell Li during her Civic Digital Fellowship with the Consumer Expenditure (CE) program at the Bureau of Labor Statistics in the summer of 2019. Minor edits made by Brandon Kopp and Alex Measure to make it suitable for sharing (see Code Updates section).

Only the frontend to the autocoder is functional in this release due to privacy issues with sharing trained models.

More information about the CE Diary Autocoder and its intended use in [this presentation]("presentation/CE Diary Autocoder.pdf").

## Code Updates

The following files are provided as examples, but are greatly reduced from their original size.

1. `ce_diary_autocoder/col_text/efdb_column_names.txt`
2. `ce_diary_autocoder/col_text/eoth_column_names.txt`
3. `ce_diary_autocoder/json/diary_key.json`
4. `ce_diary_autocoder/json/spellchecker2.json`

For disclosure and security reasons, the following models and vectorizers are not included with the repository but are referenced in the code.

1. `ce_diary_autocoder/models/emls_model.pkl`
2. `ce_diary_autocoder/models/emls_error_model.pkl`
3. `ce_diary_autocoder/models/eclo_model.pkl`
4. `ce_diary_autocoder/models/eoth_model.pkl`
5. `ce_diary_autocoder/models/efdb_model.pkl`
6. `ce_diary_autocoder/vectorizers/emls_error_vectorizer.pkl`
7. `ce_diary_autocoder/vectorizers/eclo_vectorizer.pkl`
8. `ce_diary_autocoder/vectorizers/eoth_vectorizer.pkl`
9. `ce_diary_autocoder/vectorizers/efdb_vectorizer.pkl`

## Installation

From the command prompt:
`pip install <path_to_diary_autocoder_folder>`

## Usage

From the command prompt:
`launch_diary_autocoder`

Once launched, the application can be accessed by navigating to http://127.0.0.1:8050 with a suitable web browser.
