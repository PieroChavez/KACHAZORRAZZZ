import sqlite3

import pathlib; db_path = pathlib.Path(__file__).resolve().parent.parent.parent / 'data' / 'db' / 'meta_learning.db'
conn = sqlite3.connect(str(db_path))

rows = conn.execute('SELECT id, symbol, direction, entry_price, exit_price, profit, exit_reason, regime, primary_pattern FROM trade_records ORDER BY id').fetchall()
print('=== ALL TRADE RECORDS ===')
seen = set()
unique_trades = []
dup_count = 0
for r in rows:
    key = (r[1], r[2], round(r[3],2), round(r[4],2))
    if key in seen:
        dup_count += 1
    else:
        seen.add(key)
        unique_trades.append(r)
    print(f'  id={r[0]} {r[1]} {r[2]} entry={r[3]} exit={r[4]} profit=${r[5]:+.2f} reason={r[6]} regime={r[7]} pattern={r[8]}')
print(f'\nTotal records: {len(rows)}, Duplicates: {dup_count}, Unique: {len(unique_trades)}')

print('\n=== UNIQUE TRADES SUMMARY ===')
wins = [r for r in unique_trades if r[5] > 0]
losses = [r for r in unique_trades if r[5] <= 0]
total_pnl = sum(r[5] for r in unique_trades)
print(f'Wins: {len(wins)}, Losses: {len(losses)}')
print(f'Win Rate: {len(wins)/len(unique_trades)*100:.1f}%')
print(f'Total P&L: ${total_pnl:.2f}')
if wins:
    print(f'Avg Win: ${sum(r[5] for r in wins)/len(wins):.2f}')
if losses:
    print(f'Avg Loss: ${sum(r[5] for r in losses)/len(losses):.2f}')
total_loss_abs = sum(abs(r[5]) for r in losses)
if total_loss_abs > 0:
    profit_factor = sum(r[5] for r in wins) / total_loss_abs
    print(f'Profit Factor: {profit_factor:.2f}')
if wins:
    print(f'Max Win: ${max(r[5] for r in wins):.2f}')
if losses:
    print(f'Max Loss: ${min(r[5] for r in losses):.2f}')

skipped = conn.execute('SELECT reason, COUNT(*) as cnt FROM skipped_signals GROUP BY reason ORDER BY cnt DESC').fetchall()
print('\n=== SKIPPED SIGNALS ANALYSIS ===')
print('Skipped by reason:')
for r in skipped:
    print(f'  {r[0]}: {r[1]}')
total_skipped = conn.execute('SELECT COUNT(*) FROM skipped_signals').fetchone()[0]
print(f'Total skipped signals: {total_skipped}')

patterns = conn.execute('SELECT pattern_type, total_occurrences, wins, losses, total_profit FROM pattern_performance ORDER BY total_profit DESC').fetchall()
print('\n=== PATTERN PERFORMANCE ===')
for p in patterns:
    print(f'  {p[0]}: {p[1]} occ, {p[2]}W/{p[3]}L, ${p[4]:+.2f}')

conn.close()
