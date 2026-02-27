# Invoice Classification Agent — Design Document

## What Are We Building?

We have a **Chart of Accounts (COA)** — a tree structure of financial ledger categories. When an invoice comes in, someone needs to assign it to the correct ledger (account). Doing this manually is tedious and error-prone.

We are building an **AI agent** that reads the invoice description and automatically navigates the COA tree to find the most appropriate ledger to assign the invoice to.

---

## The Tree Structure (Quick Primer)

The COA is organized like a folder hierarchy. Some nodes are **folders** (they contain more nodes inside), and some are **files** (they are the final ledger names — we call these **leaf nodes**).

```
Expenses
├── Purchase Accounts              ← LEAF (final ledger)
├── Depreciation
│   ├── Depreciation on Machinery  ← LEAF
│   └── Depreciation on Vehicles   ← LEAF
├── Direct Expense
│   ├── Raw Material Cost          ← LEAF
│   └── Labour Cost                ← LEAF
└── Indirect Expense
    └── Travelling Expense
        ├── Local Conveyance       ← LEAF
        └── Traveling Expense at Site  ← LEAF  ✅ (correct answer for a site travel invoice)
```

The agent's job is to start at the top ("Expenses") and navigate down to the correct **leaf node**.

---

## The Core Challenge

### Why not just show the agent all leaf nodes at once?

We tried that. When an agent has to pick from more than 10–15 options simultaneously, it frequently picks the wrong one. The choices become too similar and the agent gets confused.

### The Solution: Navigate Level by Level

Instead of showing the agent everything at once, we let it **explore the tree step by step** — just like a human would browse folders on a computer. At each step, the agent sees only the immediate children of the current node and decides where to go next.

This keeps the number of options small and manageable at every decision point.

---

## The Complication: Wrong Turns

Here's the tricky part. The agent might make a wrong turn.

**Example:**

Invoice description:
> `"16/05/2025 Local travel at site - Hotel to Site To and fro - AMC/SAS Site Visit"`

The agent is at the top level and sees:
- `Purchase Accounts` (leaf)
- `Depreciation` (has children)
- `Direct Expense` (has children)
- `Indirect Expense` (has children)

The agent thinks "this looks like a direct expense" and goes into **Direct Expense**.

But after exploring Direct Expense's children, it finds nothing about travel. The correct answer was actually under **Indirect Expense → Travelling Expense → Traveling Expense at Site**.

So the agent needs to be able to:
1. **Realize it went down the wrong path**
2. **Back out** of Direct Expense
3. **Try Indirect Expense** instead
4. **Find the correct leaf** there

---

## Why We Need Memory

Without memory, the agent has no idea what it has already tried. It might:
- Re-explore the same dead ends in circles
- Forget that it already ruled out certain branches
- Lose track of where it is in the tree

**Memory is what gives the agent awareness of its own exploration journey.**

---

## What Does Memory Store?

### 1. The Invoice String
The original invoice description — kept in memory so the agent always has context for every decision it makes.

### 2. Current Position (Where Am I?)
The agent always knows exactly where it is in the tree, as a full path:
```
Expenses → Indirect Expense → Travelling Expense
```

### 3. Node States (What Have I Done With Each Node?)

Every node the agent has *seen or visited* gets a label:

| State | Meaning |
|---|---|
| `UNEXPLORED` | I've seen this node as an option but haven't gone into it yet |
| `IN_PROGRESS` | I am currently inside this node, still searching |
| `EXHAUSTED` | I went in, explored fully, found nothing suitable, backed out |
| `DISCARDED` | I decided by the node's name alone that it's irrelevant — didn't even enter it |
| `SELECTED` | This is the final answer (only applies to leaf nodes) |

> **Note:** `SELECTED` only applies to leaf nodes. All other states apply to non-leaf (folder) nodes.

### 4. Exploration Log (What Have I Done?)
A running history of every action the agent has taken:
```
- Entered: Direct Expense
- Discarded: Manufacturing Expense  (reason: "invoice is about travel, not manufacturing")
- Exhausted: Direct Expense  (nothing suitable found)
- Entered: Indirect Expense
- Entered: Travelling Expense
- Selected: Traveling Expense at Site
```

This log is useful for debugging, auditing, and helping the agent reason about what's left to try.

### 5. Sibling Queue (What's Left to Try at Each Level?)
When the agent is at a node and sees multiple children, it should note which ones it hasn't tried yet. If it backtracks, it knows exactly which siblings remain to be explored — it doesn't need to re-evaluate from scratch.


---

## A Walk-Through Example

**Invoice:** `"Local travel at site - Hotel to Site"`

```
1. Agent starts at: Expenses
   Children: Purchase Accounts (LEAF), Depreciation, Direct Expense, Indirect Expense

2. Agent discards: Manufacturing Expense  → "Not relevant, travel invoice"
   Agent discards: Purchase Accounts  → "Not a purchase, it's a travel expense"

3. Agent enters: Direct Expense
   Explores children... finds Raw Material Cost, Labour Cost
   None are relevant.

4. Agent marks Direct Expense as EXHAUSTED, backs up to Expenses

5. Agent enters: Indirect Expense
   Children: Travelling Expense, Office Expense, ...

6. Agent enters: Travelling Expense
   Children: Local Conveyance, Traveling Expense at Site, ...

7. Agent selects: "Traveling Expense at Site" ✅
```

At step 4, without memory, the agent might try Direct Expense again. With memory, it knows that's already exhausted and goes straight to the next option.

---

## Summary

| Component | Purpose |
|---|---|
| Memory | Gives the agent awareness of what it has explored |
| Node States | Prevents re-exploring dead ends |
| Navigation Tools | Lets the agent move through the tree deliberately |
| Discard Tool | Lets the agent skip obviously irrelevant branches quickly |
| Exploration Log | Full audit trail of decisions made |
| Path Tracking | Agent always knows exactly where it is |

The overall goal is a **smart, self-aware agent** that narrows down from hundreds of possible ledgers to the single most appropriate one — efficiently, without confusion, and with the ability to recover from wrong turns.