{{/* SPDX-FileCopyrightText: Copyright (c) 2026 Everserve */}}
{{/* SPDX-License-Identifier: Apache-2.0 */}}
{{/* The release's name for its objects: the release name, or <release>-everserve when the
release is not named for the chart. Two releases in one cluster need two names, as with any
chart; cluster-scoped objects (the ClusterRoles) carry it too. */}}
{{- define "everserve.fullname" -}}
{{- if contains "everserve" .Release.Name -}}
{{ .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else -}}
{{ printf "%s-everserve" .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end -}}
{{- end -}}

{{/* The label the lease holder sets on its own pod, and the coordinator Service selects by:
daemons, operators and node-cert reach the holder, never a standby (templates/coordinator.yaml). */}}
{{- define "everserve.holderLabel" -}}
everserve.ai/lease-holder
{{- end -}}

{{- define "everserve.labels" -}}
app.kubernetes.io/name: everserve
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/* The image: the chart's appVersion is the tag unless one is set, as a release publishes
image and chart at one version (.github/workflows/release.yml). */}}
{{- define "everserve.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end -}}

{{/* node-cert's arguments: this node's certificate, from the coordinator's issuer (0037) */}}
{{- define "everserve.nodeCertArgs" -}}
- node-cert
- --coordinator-api
- "{{ include "everserve.fullname" . }}-coordinator.{{ .Release.Namespace }}.svc:{{ .Values.coordinator.apiPort }}"
- --coordinator-name
- {{ .Values.tls.coordinatorName | quote }}
- --ca
- /etc/everserve/ca/ca.crt
- --token-file
- /var/run/secrets/everserve/token
- --out
- /etc/everserve/node
{{- end -}}

{{- define "everserve.nodeCertMounts" -}}
- {name: node-certificate, mountPath: /etc/everserve/node}
- {name: deployment-ca, mountPath: /etc/everserve/ca, readOnly: true}
- {name: token, mountPath: /var/run/secrets/everserve, readOnly: true}
{{- end -}}

{{/* the daemon's and the store's TLS material: the files node-cert writes and renews */}}
{{- define "everserve.nodeTls" -}}
- name: EVERSERVE_TLS_CERT
  value: /etc/everserve/node/node.crt
- name: EVERSERVE_TLS_KEY
  value: /etc/everserve/node/node.key
- name: EVERSERVE_TLS_CA
  value: /etc/everserve/node/ca.crt
{{- end -}}

{{- define "everserve.nodeContainerSecurity" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities: {drop: [ALL]}
{{- end -}}
