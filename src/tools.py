"""
QuickWhistle — Phase 9 (HW4): metric <-> imperial converter via tool calling.

A pure conversion function is exposed to the LLM as a callable tool so the model
invokes it instead of doing the arithmetic itself. Flow for a question like
"How wide is an IIHF rink in feet?":

    retrieve IIHF rink rule  ->  context says the width in metres
    -> the model calls convert_units(30, "meters", "feet")
    -> 98.43 ft  ->  grounded, cited answer

We use Gemini's *automatic function calling*: the SDK reads convert_units'
type hints + docstring to build the tool declaration, calls the function when
the model requests it, and feeds the result back — no manual call loop. Every
invocation is recorded in `TOOL_CALLS` so callers/tests can verify the tool
actually fired (not the model guessing).

The verbatim system prompt is unchanged; the tool's own description drives use.

Tool calling is provider-specific; this wires the active default (Gemini).
Ollama tool-calling is possible but not wired here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.answer import build_user_message, format_context, load_system_prompt  # noqa: E402
from src.retrieve import retrieve  # noqa: E402


# ===========================================================================
# Unit tables (everything normalized to a base unit)
# ===========================================================================
# Length -> metres
_LENGTH = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "centimetre": 0.01, "centimetres": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "millimetre": 0.001, "millimetres": 0.001,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
}
# Mass -> grams
_MASS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "oz": 28.349523125, "ounce": 28.349523125, "ounces": 28.349523125,
    "lb": 453.59237, "lbs": 453.59237, "pound": 453.59237, "pounds": 453.59237,
}

# Records every convert_units invocation (args + result) for verification.
TOOL_CALLS: list[dict] = []


def _norm_unit(u: str) -> str:
    return u.strip().lower().rstrip(".").replace('"', "in").replace("'", "ft")


def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert a physical measurement between metric and imperial units.

    Supports length (meters, centimeters, millimeters, kilometers, feet, inches,
    yards) and mass (grams, kilograms, ounces, pounds). Use this tool for ANY
    unit conversion of hockey rink, goal, stick, or puck dimensions instead of
    computing the result yourself.

    Args:
        value: The numeric quantity to convert.
        from_unit: The source unit, e.g. "meters".
        to_unit: The target unit, e.g. "feet".

    Returns:
        A dict with the converted value and a human-readable summary.
    """
    f, t = _norm_unit(from_unit), _norm_unit(to_unit)
    for table in (_LENGTH, _MASS):
        if f in table and t in table:
            result = round(value * table[f] / table[t], 4)
            payload = {
                "input_value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "result": result,
                "summary": f"{value} {from_unit} = {result} {to_unit}",
            }
            TOOL_CALLS.append(payload)
            return payload
    raise ValueError(
        f"Unsupported or mismatched units: {from_unit!r} -> {to_unit!r}. "
        f"Use length or mass units (not mixed)."
    )


# ===========================================================================
# Gemini tool-calling answer
# ===========================================================================
def answer_with_tool(question: str, k: int = config.TOP_K) -> dict:
    """Answer a question with the converter tool available to the model.

    Returns {"answer", "chunks", "tool_calls"}. `tool_calls` is non-empty when
    the model actually invoked the converter.
    """
    if config.MODEL_PROVIDER != "gemini":
        raise RuntimeError(
            "answer_with_tool wires Gemini's function calling. Set "
            "MODEL_PROVIDER=gemini (Ollama tool-calling is not wired here)."
        )
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)

    chunks = retrieve(question, k=k)
    context = format_context(chunks)
    # Operational tool directive lives in the USER turn so the verbatim system
    # prompt stays unchanged. It tells the model to actually convert (via the
    # tool) rather than just offer to.
    tool_directive = (
        "\n\n[Tool use] If the user asks for a dimension in a unit different "
        "from the one in the retrieved rulebook text, you MUST call the "
        "convert_units tool to perform the conversion — do not compute it "
        "yourself and do not merely offer to convert. If a dimension is a range "
        "(e.g. 26-30 m), convert both endpoints. Then give the converted "
        "value(s) and cite the rule."
    )
    user_message = build_user_message(question, context) + tool_directive

    model = genai.GenerativeModel(
        config.GEMINI_MODEL,
        system_instruction=load_system_prompt(),
        tools=[convert_units],  # SDK builds the declaration from hints+docstring
    )
    chat = model.start_chat(enable_automatic_function_calling=True)

    TOOL_CALLS.clear()
    resp = chat.send_message(
        user_message,
        generation_config={
            "temperature": config.TEMPERATURE,
            "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        },
    )
    return {
        "answer": resp.text.strip(),
        "chunks": chunks,
        "tool_calls": list(TOOL_CALLS),
    }


# ===========================================================================
# Self-check
# ===========================================================================
def _offline_tests() -> None:
    """Verify the pure converter (no LLM)."""
    import math

    cases = [
        (convert_units(30, "meters", "feet")["result"], 98.4252),
        (convert_units(85, "feet", "meters")["result"], 25.908),
        (convert_units(1, "inch", "cm")["result"], 2.54),
        (convert_units(170, "grams", "ounces")["result"], 5.9966),
        (convert_units(6, "ounces", "grams")["result"], 170.0971),
    ]
    print("Offline converter checks:")
    ok = True
    for got, exp in cases:
        good = math.isclose(got, exp, rel_tol=1e-3)
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {got} ~= {exp}")
    # mismatched dimensions should raise
    try:
        convert_units(1, "meters", "grams")
        print("  [FAIL] mismatched units did not raise")
        ok = False
    except ValueError:
        print("  [PASS] mismatched units raised ValueError")
    print("OFFLINE:", "ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    _offline_tests()
    if "--live" in sys.argv:
        print("\nLive tool-calling demo (Gemini):")
        for q in ["How wide is an IIHF rink in feet?"]:
            res = answer_with_tool(q)
            print(f"\nQ: {q}")
            print(f"tool_calls: {res['tool_calls']}")
            print(f"answer:\n{res['answer']}")
