# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The commit this tree is, stamped by ``git archive`` (``export-subst`` in ``.gitattributes``).

In a checkout the placeholder stays, and :func:`everserve_vllm.identity.source_commit` asks git.
"""

COMMIT = "$Format:%H$"
