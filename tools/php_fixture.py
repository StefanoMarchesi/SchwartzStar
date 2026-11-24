"""Small, deterministic PHP fixture for the Gemma/Inkling coding probe.

The fixture deliberately has no Composer or Git dependency.  ``populate`` is
given an already-created repository directory and only creates the source and
test files below it.
"""

from pathlib import Path


FRAGMENTS = (
    "Invoice amounts are integer cents; subtotal multiplies each item's unit_cents by quantity.",
    "Subtract discount before tax; tax_rate_bps is basis points and tax is floor(intdiv(taxable_cents * tax_rate_bps, 10000)).",
    "The sample has 3x1200 and 2x875 cents, discount 400, rate 875 bps: subtotal 5350, tax 433, total 5383; turn 2 applies a 600-cent credit after tax and never pays below zero.",
)

SHORT_CONTEXT = " ".join(FRAGMENTS)

TURN_1 = """Work in this PHP repository without Composer or network access. Implement the invoice/cost contract described in the supplied context. In the first turn, edit both src/invoice.php and src/costs.php: make the invoice breakdown calculate integer-cent subtotal, discount-before-tax, floor basis-point tax, and total, and make the cost summary expose the same values. Preserve the existing public function names. Run `php tests/run.php 1` and leave the repository with that test passing. Do not edit tests/run.php."""

TURN_2 = """Continue in the same PHP repository after turn 1. Add customer-credit support in both source modules while preserving every turn-1 API. In src/invoice.php add invoice_breakdown_with_credit_cents($items, $discount_cents, $tax_rate_bps, $credit_cents): credit is applied after tax and payable must never be negative. In src/costs.php add cost_summary_with_credit_cents with the same inputs. Both new functions return subtotal_cents, discount_cents, tax_cents, total_cents before credit, credit_cents, and payable_cents, in that order. Reuse the turn-1 calculation instead of duplicating arithmetic. Run `php tests/run.php 2` and leave that test passing. Do not edit tests/run.php."""

TEST_COMMANDS = ("php tests/run.php 1", "php tests/run.php 2")


_INVOICE_PHP = r'''<?php

function invoice_subtotal_cents(array $items): int
{
    $subtotal = 0;
    foreach ($items as $item) {
        $subtotal += (int) ($item['unit_cents'] ?? 0) * (int) ($item['quantity'] ?? 0);
    }
    return $subtotal;
}

function invoice_breakdown(array $items, int $discount_cents, int $tax_rate_bps): array
{
    $subtotal = invoice_subtotal_cents($items);
    $discount = max(0, min($discount_cents, $subtotal));
    $taxable = $subtotal - $discount;

    // Deliberately incomplete for turn 1: the agent must implement the contract.
    $tax = 0;
    $total = $taxable;

    return [
        'subtotal_cents' => $subtotal,
        'discount_cents' => $discount,
        'tax_cents' => $tax,
        'total_cents' => $total,
    ];
}

function invoice_total_cents(array $items, int $discount_cents, int $tax_rate_bps): int
{
    return invoice_breakdown($items, $discount_cents, $tax_rate_bps)['total_cents'];
}
'''


_COSTS_PHP = r'''<?php

function cost_summary_cents(array $breakdown): array
{
    // Deliberately mirrors the incomplete initial invoice result for turn 1.
    return [
        'subtotal_cents' => (int) ($breakdown['subtotal_cents'] ?? 0),
        'discount_cents' => (int) ($breakdown['discount_cents'] ?? 0),
        'tax_cents' => 0,
        'total_cents' => (int) ($breakdown['total_cents'] ?? 0),
    ];
}
'''


_RUN_PHP = r'''<?php

declare(strict_types=1);

require_once __DIR__ . '/../src/invoice.php';
require_once __DIR__ . '/../src/costs.php';

function fail_test(string $message): never
{
    fwrite(STDERR, "FAIL: {$message}\n");
    exit(1);
}

function assert_same(mixed $expected, mixed $actual, string $message): void
{
    // Keep failure text useful without printing the whole contract into the prompt.
    if ($expected !== $actual) {
        fail_test($message);
    }
}

function sample_items(): array
{
    return [
        ['unit_cents' => 1200, 'quantity' => 3],
        ['unit_cents' => 875, 'quantity' => 2],
    ];
}

function expected_turn_one(): array
{
    $subtotal = 3 * 1200 + 2 * 875;
    $discount = 400;
    $tax = intdiv(($subtotal - $discount) * 875, 10000);
    return [
        'subtotal_cents' => $subtotal,
        'discount_cents' => $discount,
        'tax_cents' => $tax,
        'total_cents' => $subtotal - $discount + $tax,
    ];
}

$turn = $argv[1] ?? '';
$items = sample_items();
$breakdown = invoice_breakdown($items, 400, 875);
$expected = expected_turn_one();

if ($turn === '1') {
    assert_same($expected, $breakdown, 'invoice breakdown contract');
    assert_same($expected, cost_summary_cents($breakdown), 'cost summary contract');
    assert_same($expected['total_cents'], invoice_total_cents($items, 400, 875), 'invoice total contract');
    fwrite(STDOUT, "turn 1 ok\n");
    exit(0);
}

if ($turn === '2') {
    if (!function_exists('invoice_breakdown_with_credit_cents')
        || !function_exists('cost_summary_with_credit_cents')) {
        fail_test('credit API is missing');
    }

    $with_credit = invoice_breakdown_with_credit_cents($items, 400, 875, 600);
    $expected_with_credit = $expected;
    $expected_with_credit['credit_cents'] = 600;
    $expected_with_credit['payable_cents'] = max(0, $expected['total_cents'] - 600);
    assert_same($expected_with_credit, $with_credit, 'credit invoice contract');
    assert_same($expected_with_credit, cost_summary_with_credit_cents($items, 400, 875, 600), 'credit cost contract');

    $zero = invoice_breakdown_with_credit_cents($items, 400, 875, 999999);
    assert_same(0, $zero['payable_cents'], 'credit floor contract');
    fwrite(STDOUT, "turn 2 ok\n");
    exit(0);
}

fail_test('usage: php tests/run.php 1|2');
'''


def populate(root: Path) -> None:
    """Write the initial fixture into an existing repository directory."""

    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"fixture root must already exist: {root}")

    src = root / "src"
    tests = root / "tests"
    src.mkdir(exist_ok=True)
    tests.mkdir(exist_ok=True)
    (src / "invoice.php").write_text(_INVOICE_PHP, encoding="utf-8")
    (src / "costs.php").write_text(_COSTS_PHP, encoding="utf-8")
    (tests / "run.php").write_text(_RUN_PHP, encoding="utf-8")


__all__ = [
    "FRAGMENTS",
    "SHORT_CONTEXT",
    "TEST_COMMANDS",
    "TURN_1",
    "TURN_2",
    "populate",
]
