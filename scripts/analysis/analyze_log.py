import re, sys

for log_file in sys.argv[1:]:
    print(f'\n{"="*70}')
    print(f'LOG: {log_file}')
    print(f'{"="*70}')
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print('  FILE NOT FOUND')
        continue

    wins = 0; losses = 0; win_total = 0.0; loss_total = 0.0
    be_count = 0; trail_count = 0; scale_out_count = 0
    errors = []; trade_lines = []; entries = []
    sl_hits = 0; tp_hits = 0

    for line in lines:
        if 'P/L=' in line:
            m = re.search(r'P/L=\s*\$?([-]?\d+\.?\d*)', line)
            if m:
                val = float(m.group(1))
                if val > 0: wins += 1; win_total += val
                else: losses += 1; loss_total += abs(val)
        if 'BE' in line and ('activ' in line.lower()):
            be_count += 1
        if 'trail' in line.lower() and ('stop' in line.lower() or 'activ' in line.lower()):
            trail_count += 1
        if 'scale_out' in line or 'Scale-out' in line:
            scale_out_count += 1
        if 'SL ' in line and ('hit' in line.lower() or 'reach' in line.lower() or 'activ' in line.lower()):
            sl_hits += 1
        if 'TP ' in line and ('hit' in line.lower() or 'reach' in line.lower() or 'activ' in line.lower()):
            tp_hits += 1
        if 'ERROR' in line:
            errors.append(line.strip())
        if 'Trade opened' in line or 'Entry' in line:
            entries.append(line.strip())
        if 'P/L=' in line or 'Score:' in line or 'Skipped' in line or 'Trade' in line:
            trade_lines.append(line.strip())

    total_trades = wins + losses
    print(f'Wins: {wins}, Losses: {losses}')
    if total_trades > 0:
        print(f'Win Rate: {wins/total_trades*100:.1f}%')
    print(f'Total Win: ${win_total:.2f}')
    print(f'Total Loss: ${loss_total:.2f}')
    print(f'Net P&L: ${win_total - loss_total:.2f}')
    if loss_total > 0:
        print(f'Profit Factor: {win_total/loss_total:.2f}')
    print(f'BE activations: {be_count}')
    print(f'Trail events: {trail_count}')
    print(f'Scale-out events: {scale_out_count}')
    print(f'SL hits: {sl_hits}, TP hits: {tp_hits}')
    print(f'Entries/Openings: {len(entries)}')

    score_lines = [l for l in lines if 'Score:' in l]
    skipped = [l for l in lines if 'Skipped' in l]
    print(f'Score evaluations: {len(score_lines)}')
    print(f'Skipped signals: {len(skipped)}')
    print(f'\nERROR lines: {len(errors)}')
    for e in errors[:5]:
        print(f'  {e}')
    print(f'Total lines: {len(lines)}')
