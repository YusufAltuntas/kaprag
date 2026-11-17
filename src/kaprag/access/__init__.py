"""Role-based access control over the chunk index: a SIMULATED access model.

The corpus (bank annual reports) is public. The roles here are assigned by a fixed
rule purely to demonstrate the mechanism an enterprise RAG system needs: content a
user may not see is never retrieved, because the filter is part of the SQL query of
every retrieval arm, before ranking.

- `kaprag.access.roles`: the role vocabulary, a request's effective roles, the role
  sets a chunk can carry and their precedence.
- `kaprag.access.rules`: the indexing-time rule (headings and statutory report
  sections of the parsed Docling rows -> a chunk's role set).
- `kaprag.access.filter`: the SQL condition for a request's role.
- `kaprag.access.leak`: the deterministic leak test's queries and leak counting.
"""
