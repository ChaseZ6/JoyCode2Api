import React from 'react';
import { Progress, Tag, Typography } from 'antd';
import type { Stats } from '../api';

const MONTHLY_TOKEN_LIMIT = 1_000_000_000;
const fmt = (n: number) => (n / 1_000_000).toFixed(2);

const MonthlyBudget: React.FC<{ usage: Stats['month_to_date']; todayTokens: number }> = ({ usage, todayTokens }) => {
  const monthStart = Date.parse(usage.month_start);
  const monthEnd = Date.parse(usage.month_end);
  const asOf = Date.parse(usage.as_of);
  if (!Number.isFinite(monthStart) || !Number.isFinite(monthEnd)
    || !Number.isFinite(asOf) || monthEnd <= monthStart) return null;

  const usedTokens = usage.total_input_tokens + usage.total_output_tokens;
  const remainingTokens = Math.max(MONTHLY_TOKEN_LIMIT - usedTokens, 0);
  const remainingQuotaPercent = remainingTokens / MONTHLY_TOKEN_LIMIT * 100;
  const remainingTime = Math.max(0, Math.min(monthEnd - asOf, monthEnd - monthStart));
  const remainingTimePercent = remainingTime / (monthEnd - monthStart) * 100;
  const remainingDays = remainingTime / 86_400_000;
  const overBudget = usedTokens > MONTHLY_TOKEN_LIMIT;
  const quotaExhausted = usedTokens >= MONTHLY_TOKEN_LIMIT;
  const usageAhead = remainingQuotaPercent < remainingTimePercent;

  return (
    <div className="jc-month-budget">
      <div className="jc-month-budget-summary">
        <div className="jc-month-budget-metrics">
          <div>
            <div className="jc-month-budget-caption">今日已用</div>
            <div className="jc-month-budget-value jc-month-budget-today">{fmt(todayTokens)}<span>M</span></div>
          </div>
          <div>
            <div className="jc-month-budget-caption">{overBudget ? '本月已超额' : '本月剩余'}</div>
            <div className={`jc-month-budget-value${overBudget ? ' jc-month-budget-over' : ''}`}>
              {fmt(overBudget ? usedTokens - MONTHLY_TOKEN_LIMIT : remainingTokens)}<span>M</span>
            </div>
          </div>
        </div>
        <div className="jc-month-budget-meta">
          <Typography.Text type="secondary">月度上限 1000M · 全部账号</Typography.Text>
          {quotaExhausted ? <Tag color="error">额度已用完</Tag>
            : usageAhead ? <Tag color="warning">使用偏快</Tag> : null}
        </div>
      </div>
      <div className="jc-month-budget-bars">
        <div>
          <div className="jc-month-budget-label">
            <Typography.Text>额度剩余</Typography.Text>
            <Typography.Text className="jc-mono">{remainingQuotaPercent.toFixed(1)}%</Typography.Text>
          </div>
          <div role="progressbar" aria-label="本月额度剩余" aria-valuenow={remainingQuotaPercent} aria-valuemin={0} aria-valuemax={100}>
            <Progress percent={remainingQuotaPercent} showInfo={false}
              size="small" strokeColor={quotaExhausted ? 'var(--jc-danger)' : usageAhead ? 'var(--jc-warning)' : 'var(--jc-accent)'} />
          </div>
        </div>
        <div>
          <div className="jc-month-budget-label">
            <Typography.Text>时间剩余 <Typography.Text type="secondary">· {remainingDays.toFixed(1)} 天</Typography.Text></Typography.Text>
            <Typography.Text className="jc-mono">{remainingTimePercent.toFixed(1)}%</Typography.Text>
          </div>
          <div role="progressbar" aria-label="本月时间剩余" aria-valuenow={remainingTimePercent} aria-valuemin={0} aria-valuemax={100}>
            <Progress percent={remainingTimePercent} showInfo={false} size="small" strokeColor="var(--jc-info)" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default MonthlyBudget;
