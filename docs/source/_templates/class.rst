{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :special-members: __call__, __getitem__
   :inherited-members:
{% block methods %}
{% if all_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :template: method.rst
{% for item in all_methods %}
{%- if not item.startswith('_') or item in ['__call__', '__getitem__'] %}
      ~{{ name }}.{{ item }}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{% endblock %}
{% block attributes %}
{% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :template: attribute.rst
{% for item in attributes %}
      ~{{ name }}.{{ item }}
{%- endfor %}
{%- endif %}
{% endblock %}
