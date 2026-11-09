"""Generate CULBM's collision and equilibrium operators with lbmpy."""

from pathlib import Path

import sympy as sp
from lbmpy.creationfunctions import LBMConfig, LBMOptimisation, create_lb_collision_rule
from lbmpy.enums import Method, Stencil
from pystencils import Assignment
from pystencils.simp import AssignmentCollection
from sympy.codegen.ast import float32, real
from sympy.printing.c import C99CodePrinter

OUT = Path(__file__).resolve().parent / "generated"
OMEGA = sp.Symbol("omega")
RHO, UX, UY, UZ = sp.symbols("rho_in ux_in uy_in uz_in")
DELTA_RHO = sp.Symbol("delta_rho_in")


class FloatPrinter(C99CodePrinter):
    def _print_Pow(self, value):
        base, exponent = value.args
        assert exponent.is_Integer
        n = int(exponent)
        assert n != 0
        product = " * ".join([f"({self._print(base)})"] * abs(n))
        return f"(1.0f / ({product}))" if n < 0 else f"({product})"


PRINTER = FloatPrinter({"type_aliases": {real: float32}})


def external_inputs(rule):
    assigned = {a.lhs for a in rule.all_assignments}
    referenced = set().union(*(a.rhs.free_symbols for a in rule.all_assignments))
    return referenced - assigned


def emit_rule(name, rule, mapping):
    rule = rule.new_with_substitutions(
        {
            part: next(rule.subexpression_symbol_generator)
            for assignment in rule.subexpressions
            for part in assignment.rhs.args
            if part.is_Add
        },
        add_substitutions_as_subexpressions=True,
        substitute_on_lhs=False,
    ).new_without_unused_subexpressions()
    rule.topological_sort()
    assert [a.lhs for a in rule.main_assignments] == list(rule.method.post_collision_pdf_symbols)
    lines = [f"constexpr void {name}(real_t omega, ddf_t* g)", "{"]
    for symbol, index in zip(rule.method.pre_collision_pdf_symbols, mapping, strict=True):
        lines.append(f"    const ddf_t {symbol} = g[{index}];")
    for assignment in rule.subexpressions:
        lines.append(f"    const real_t {assignment.lhs} = {PRINTER.doprint(assignment.rhs)};")
    for assignment, index in zip(rule.main_assignments, mapping, strict=True):
        lines.append(f"    g[{index}] = ({PRINTER.doprint(assignment.rhs)});")
    lines.append("}")
    return "\n".join(lines)


def emit_equilibrium(name, method, mapping):
    cqc = method.conserved_quantity_computation
    inputs = AssignmentCollection([
        Assignment(cqc.density_symbol, RHO),
        Assignment(cqc.density_deviation_symbol, DELTA_RHO),
        *[Assignment(symbol, value) for symbol, value in zip(cqc.velocity_symbols, (UX, UY, UZ), strict=True)],
    ])
    rule = method.get_equilibrium(
        conserved_quantity_equations=inputs,
        subexpressions=False,
        keep_cqc_subexpressions=False,
    )
    inputs = external_inputs(rule)
    assert inputs in ({RHO, UX, UY, UZ}, {RHO, DELTA_RHO, UX, UY, UZ})
    substitutions, expressions = sp.cse([sp.horner(a.rhs, UX, UY, UZ) for a in rule.main_assignments])
    lines = [f"constexpr void {name}(real_t rho_in, real_t ux_in, real_t uy_in, real_t uz_in, ddf_t* g)", "{"]
    if DELTA_RHO in inputs:
        lines.append(f"    const real_t {DELTA_RHO} = {PRINTER.doprint(RHO - 1)};")
    lines.extend(f"    const real_t {s} = {PRINTER.doprint(v)};" for s, v in substitutions)
    lines.extend(f"    g[{index}] = {PRINTER.doprint(expr)};" for index, expr in zip(mapping, expressions, strict=True))
    lines.append("}")
    return "\n".join(lines)


def emit_macroscopic(method, mapping):
    outputs = sp.symbols("rho_out ux_out uy_out uz_out")
    rule = method.conserved_quantity_computation.output_equations_from_pdfs(
        method.pre_collision_pdf_symbols,
        {"density": outputs[0], "velocity": tuple(outputs[1:])},
    )
    assert external_inputs(rule) == set(method.pre_collision_pdf_symbols)
    rule = rule.new_without_subexpressions()
    substitutions, expressions = sp.cse([a.rhs for a in rule.main_assignments])
    lines = ["constexpr void calc_rho_u(real_t& rho_out, real_t& ux_out, real_t& uy_out, real_t& uz_out, const ddf_t* g)", "{"]
    for symbol, index in zip(method.pre_collision_pdf_symbols, mapping, strict=True):
        lines.append(f"    const ddf_t {symbol} = g[{index}];")
    lines.extend(f"    const real_t {s} = {PRINTER.doprint(v)};" for s, v in substitutions)
    lines.extend(f"    {assignment.lhs} = {PRINTER.doprint(expr)};" for assignment, expr in zip(rule.main_assignments, expressions, strict=True))
    lines.append("}")
    return "\n".join(lines)


for q, stencil, models in (
    (15, Stencil.D3Q15, (("bgk", Method.SRT), ("mrt", Method.MRT))),
    (19, Stencil.D3Q19, (("bgk", Method.SRT), ("mrt", Method.MRT), ("cumulant", Method.CUMULANT))),
    (27, Stencil.D3Q27, (("bgk", Method.SRT), ("mrt", Method.MRT), ("cumulant", Method.CUMULANT))),
):
    directions = [
        (x, y, z)
        for z in (-1, 0, 1)
        for y in (-1, 0, 1)
        for x in (-1, 0, 1)
        if q == 27
        or (q == 19 and abs(x) + abs(y) + abs(z) <= 2)
        or (q == 15 and abs(x) + abs(y) + abs(z) != 2)
    ]
    assert len(directions) == q
    assert all(directions[q - 1 - i] == tuple(-c for c in directions[i]) for i in range(q))
    production = []
    for name, method_kind in models:
        config = LBMConfig(
            stencil=stencil,
            method=method_kind,
            relaxation_rate=OMEGA,
            compressible=True,
            zero_centered=True,
            weighted=False,
            continuous_equilibrium=False,
            equilibrium_order=2,
            force=(0, 0, 0),
            galilean_correction=False,
            fourth_order_correction=False,
        )
        rule = create_lb_collision_rule(
            lbm_config=config,
            lbm_optimisation=LBMOptimisation(cse_global=True, cse_pdfs=True),
        )
        method = rule.method
        mapping = [directions.index(tuple(v)) for v in method.stencil]
        assert sorted(mapping) == list(range(q))
        assert external_inputs(rule) == set(method.pre_collision_pdf_symbols) | {OMEGA}
        assert method.force_model is None
        production.extend([
            emit_rule(f"{name}_collide", rule, mapping),
            emit_equilibrium(f"{name}_equilibrium", method, mapping),
        ])
    production.append(emit_macroscopic(method, mapping))
    prefix = "// Generated by src/core/collision/generate.py; do not edit.\n#pragma once\n#include \"core/types.hpp\"\n\n"
    content = (
        prefix
        + f"namespace culbm::core::collision::generated::d3q{q}\n{{\n"
        + "\n\n".join(production)
        + "\n}\n"
    )
    (OUT / f"d3q{q}.cuh").write_text(content)
