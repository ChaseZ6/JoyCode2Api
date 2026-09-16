import React, { useEffect, useRef, useState } from 'react';
import {
  Card, Row, Col, Statistic, Typography, Skeleton, Tag, Select, Button, Input,
  message, Space, Table, Badge, Segmented, Popconfirm, Tooltip, Divider,
} from 'antd';
import type { TableColumnsType } from 'antd';
import {
  ArrowLeftOutlined, ApiOutlined, ThunderboltOutlined,
  CheckCircleOutlined, ReloadOutlined,
  ClockCircleOutlined, GlobalOutlined, FireOutlined,
  DeleteOutlined, QuestionCircleOutlined, InfoCircleOutlined,
  CloseCircleOutlined, SwapOutlined, CopyOutlined,
  DownOutlined, UpOutlined,
} from '@ant-design/icons';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
  ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area,
} from 'recharts';
import { useParams, useNavigate } from 'react-router-dom';
import { api, accountDisplayName } from '../api';
import type { Account, AccountStats, ModelInfo, RequestLog } from '../api';
import { writeClipboardText } from '../utils/clipboard';

// Offline fallback only. The selector is normally populated from
// /api/accounts/:id/models, which reflects what the account can actually use.
const BUILTIN_MODELS = [
  { label: 'JoyAI-Code-1.5（推荐）', value: 'JoyAI-Code-1.5' },
  { label: 'GPT-6 Astra', value: 'gpt-6-astra' },
];

const isClaudeModel = (model?: string) => !!model && /^claude/i.test(model);

const PIE_COLORS = ['#22C55E', '#3B82F6', '#F59E0B', '#EF4444', '#A855F7', '#06B6D4', '#EC4899', '#84CC16'];

const CHART_COLORS = {
  primary: '#22C55E',
  secondary: '#3B82F6',
  danger: '#EF4444',
  warning: '#F59E0B',
  grid: '#1F2937',
  axis: '#475569',
};

const latencyColor = (ms: number) => {
  if (ms < 500) return '#22C55E';
  if (ms < 1500) return '#F59E0B';
  return '#EF4444';
};

const fmtTokens = (n: number) => {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n.toLocaleString();
};

const statusTag = (code: number) => {
  if (code >= 200 && code < 300) return <Tag color="success">{code}</Tag>;
  if (code >= 400 && code < 500) return <Tag color="warning">{code}</Tag>;
  return <Tag color="error">{code}</Tag>;
};

const formatTime = (t: string) => {
  if (!t) return '-';
  // Request timestamps are stored by SQLite as server-local wall-clock text
  // (YYYY-MM-DD HH:mm:ss). Appending Z would falsely label them as UTC and
  // shift the displayed time by the browser's timezone offset.
  const normalized = t.replace('T', ' ');
  return normalized.length >= 19 ? normalized.slice(0, 19) : normalized;
};

const formatLatency = (ms: number) => {
  if (ms < 1000) return `${ms}ms`;
  const s = Math.floor(ms / 1000);
  const remainMs = ms % 1000;
  if (s < 60) return `${s}s${remainMs > 0 ? ` ${remainMs}ms` : ''}`;
  const m = Math.floor(s / 60);
  const remainS = s % 60;
  return `${m}m${remainS > 0 ? ` ${remainS}s` : ''}`;
};

interface LogDetailField {
  label: string;
  value: string;
}

const getLogDetailFields = (record: RequestLog): LogDetailField[] => [
  { label: '请求 ID', value: String(record.id) },
  { label: '时间', value: formatTime(record.created_at) },
  { label: '端点', value: record.endpoint },
  { label: '模型', value: record.model || '-' },
  { label: '流式', value: record.stream ? '是' : '否' },
  { label: '状态', value: String(record.status_code) },
  { label: '输入 / 输出 Token', value: `${record.input_tokens || 0} / ${record.output_tokens || 0}` },
  { label: '延迟', value: `${formatLatency(record.latency_ms)} (${record.latency_ms}ms)` },
];

const formatLogDetails = (record: RequestLog) => {
  const lines = [
    'JoyCode2Api 请求日志',
    ...getLogDetailFields(record).map(({ label, value }) => `${label}: ${value}`),
  ];
  if (record.error_message || record.status_code >= 400) {
    lines.push('错误详情:', record.error_message || `HTTP ${record.status_code}`);
  }
  return lines.join('\n');
};

const AccountDetail: React.FC = () => {
  const { userId } = useParams<{ userId: string }>();
  const navigate = useNavigate();
  const [account, setAccount] = useState<Account | null>(null);
  const [stats, setStats] = useState<AccountStats | null>(null);
  const [statsSummary, setStatsSummary] = useState<AccountStats | null>(null);
  const [logs, setLogs] = useState<RequestLog[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [modelLoading, setModelLoading] = useState(false);
  const [savingModel, setSavingModel] = useState(false);
  const [logFilter, setLogFilter] = useState<'all' | 'stream' | 'error'>('all');
  const [logSearchInput, setLogSearchInput] = useState('');
  const [logSearch, setLogSearch] = useState('');
  const [logPage, setLogPage] = useState(1);
  const [logPageSize, setLogPageSize] = useState(20);
  const [logTotal, setLogTotal] = useState(0);
  const [logLoading, setLogLoading] = useState(false);
  const [logAutoRefresh, setLogAutoRefresh] = useState(true);
  const logRequestSeq = useRef(0);
  const logPageRef = useRef(1);
  const [activeSessions, setActiveSessions] = useState(0);
  const [activeSessionsAutoRefresh, setActiveSessionsAutoRefresh] = useState(true);
  const [statsAutoRefresh, setStatsAutoRefresh] = useState(true);
  const [statsCollapsed, setStatsCollapsed] = useState(true);

  const decodedKey = userId ? decodeURIComponent(userId) : '';

  const fetchStats = async (summaryOnly = false) => {
    const nextStats = await api.getAccountStats(decodedKey);
    setStatsSummary(nextStats);
    if (!summaryOnly) setStats(nextStats);
  };

  const fetchData = async () => {
    setLoading(true);
    try {
      const [accounts] = await Promise.all([
        api.listAccounts(),
      ]);
      const acc = accounts.find((a) => a.user_id === decodedKey);
      setAccount(acc || null);
      await fetchStats();
    } catch (e: unknown) {
      message.error(e instanceof Error ? e.message : '加载失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchModels = async () => {
    setModelLoading(true);
    try {
      const data = await api.listAccountModels(decodedKey);
      setModels(data);
    } catch {
      // fallback to builtin
    } finally {
      setModelLoading(false);
    }
  };

  const fetchLogs = async (silent = false) => {
    const requestSeq = ++logRequestSeq.current;
    if (!silent) setLogLoading(true);
    try {
      const data = await api.getAccountLogs(decodedKey, {
        limit: logPageSize, offset: (logPage - 1) * logPageSize, kind: logFilter,
        q: logSearch || undefined,
      });
      if (requestSeq !== logRequestSeq.current) return;
      setLogs(data.logs || []);
      setLogTotal(data.total);
    } catch (e: unknown) {
      if (!silent && requestSeq === logRequestSeq.current) {
        message.error(e instanceof Error ? e.message : '请求日志加载失败');
      }
    } finally {
      if (!silent && requestSeq === logRequestSeq.current) setLogLoading(false);
    }
  };

  const handleLogSearch = (value: string) => {
    const search = value.trim();
    setLogSearchInput(search);
    setLogPage(1);
    setLogSearch(search);
  };

  const handleCopyLog = async (record: RequestLog) => {
    try {
      await writeClipboardText(formatLogDetails(record));
      message.success('日志详情已复制到剪贴板');
    } catch {
      message.error('复制失败，请检查浏览器剪贴板权限');
    }
  };

  useEffect(() => { fetchData(); }, [decodedKey]);
  useEffect(() => { fetchModels(); }, [decodedKey]);
  useEffect(() => { fetchLogs(); }, [decodedKey, logPage, logPageSize, logFilter, logSearch]);
  useEffect(() => { logPageRef.current = logPage; }, [logPage]);

  // Poll the first log page while auto-refresh is on. Silent polling keeps the
  // table stable, and skipping pages beyond the first avoids shuffling rows
  // out from under someone reading older entries.
  useEffect(() => {
    if (!logAutoRefresh) return;
    const id = setInterval(() => {
      if (logPageRef.current === 1) fetchLogs(true);
    }, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logAutoRefresh, decodedKey, logPageSize, logFilter, logSearch]);

  // Daily aggregates use the local calendar day. Refresh only the stats just
  // after local midnight so an open page resets without reloading 500 log rows.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const scheduleMidnightRefresh = () => {
      const now = new Date();
      const nextMidnight = new Date(now);
      nextMidnight.setHours(24, 0, 1, 0);
      timer = setTimeout(async () => {
        try {
          await fetchStats();
        } catch {
          // A manual refresh remains available if the midnight request fails.
        }
        if (!cancelled) scheduleMidnightRefresh();
      }, Math.max(1000, nextMidnight.getTime() - now.getTime()));
    };

    scheduleMidnightRefresh();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [decodedKey]);

  // Keep the summary current without replacing the detail snapshot.
  useEffect(() => {
    if (!stats || !statsAutoRefresh) return;
    const pollSummary = async () => {
      try {
        await fetchStats(true);
      } catch {
        // The next interval or opening details will retry the request.
      }
    };
    const id = setInterval(pollSummary, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [decodedKey, statsAutoRefresh, stats]);

  const handleStatsToggle = async () => {
    if (!statsCollapsed) {
      setStatsCollapsed(true);
      return;
    }
    try {
      await fetchStats();
    } catch (e: unknown) {
      message.error(e instanceof Error ? e.message : '统计详情刷新失败');
    }
    setStatsCollapsed(false);
  };

  // Poll active sessions every 5s
  useEffect(() => {
    if (!activeSessionsAutoRefresh) return;
    const poll = async () => {
      try {
        const accounts = await api.listAccounts();
        const acc = accounts.find((a) => a.user_id === decodedKey);
        if (acc) setActiveSessions(acc.active_sessions);
      } catch { /* ignore */ }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, [decodedKey, activeSessionsAutoRefresh]);

  const handleModelChange = async (newModel: string) => {
    setSavingModel(true);
    try {
      await api.updateAccountModel(decodedKey, newModel);
      message.success(`默认模型已更新为「${newModel || '未设置'}」`);
      fetchData();
    } catch (e: unknown) {
      message.error(e instanceof Error ? e.message : '更新失败');
    } finally {
      setSavingModel(false);
    }
  };

  if (loading) {
    return (
      <div className="jc-page">
        <Skeleton.Button active block style={{ height: 80, marginBottom: 16, borderRadius: 10 }} />
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}><Card size="small"><Skeleton active paragraph={{ rows: 5 }} /></Card></Col>
          <Col xs={24} md={12}><Card size="small"><Skeleton active paragraph={{ rows: 5 }} /></Card></Col>
        </Row>
      </div>
    );
  }
  if (!account) return <div style={{ textAlign: 'center', padding: 100 }}>账号不存在</div>;

  // Prefer live upstream models; fall back to the builtin list only if the
  // request failed. The API returns {id,name}, so both fields are populated.
  const allModelOptions = models.length > 0
    ? models.map((m) => ({ label: m.name || m.id, value: m.id }))
    : BUILTIN_MODELS;

  const endpointData = stats?.by_endpoint.map((e) => ({
    name: e.endpoint.replace('/v1/', ''),
    value: e.count,
  })) || [];

  const logColumns: TableColumnsType<RequestLog> = [
    {
      title: '时间',
      dataIndex: 'created_at',
      key: 'time',
      width: 170,
      render: (t: string) => (
        <Typography.Text style={{ fontSize: 12, fontFamily: 'monospace' }}>
          {formatTime(t)}
        </Typography.Text>
      ),
    },
    {
      title: '端点',
      dataIndex: 'endpoint',
      key: 'endpoint',
      width: 200,
      responsive: ['sm'],
      render: (ep: string) => (
        <Typography.Text code style={{ fontSize: 12 }}>{ep}</Typography.Text>
      ),
    },
    {
      title: '模型',
      dataIndex: 'model',
      key: 'model',
      width: 140,
      ellipsis: true,
      render: (m: string) => m || <Typography.Text type="secondary">-</Typography.Text>,
    },
    {
      title: '流式',
      dataIndex: 'stream',
      key: 'stream',
      width: 60,
      responsive: ['md'],
      render: (s: boolean) => s
        ? <Badge status="processing" text="" />
        : <Badge status="default" text="" />,
    },
    {
      title: '状态',
      dataIndex: 'status_code',
      key: 'status',
      width: 70,
      render: (code: number) => statusTag(code),
    },
    {
      title: '输入',
      dataIndex: 'input_tokens',
      key: 'input',
      width: 80,
      responsive: ['md'],
      sorter: (a: RequestLog, b: RequestLog) => a.input_tokens - b.input_tokens,
      render: (n: number) => (
        <Typography.Text style={{ fontSize: 12, fontFamily: 'monospace' }}>
          {n > 0 ? fmtTokens(n) : '-'}
        </Typography.Text>
      ),
    },
    {
      title: '输出',
      dataIndex: 'output_tokens',
      key: 'output',
      width: 80,
      responsive: ['md'],
      sorter: (a: RequestLog, b: RequestLog) => a.output_tokens - b.output_tokens,
      render: (n: number) => (
        <Typography.Text style={{ fontSize: 12, fontFamily: 'monospace' }}>
          {n > 0 ? fmtTokens(n) : '-'}
        </Typography.Text>
      ),
    },
    {
      title: '延迟',
      dataIndex: 'latency_ms',
      key: 'latency',
      width: 100,
      responsive: ['sm'],
      sorter: (a: RequestLog, b: RequestLog) => a.latency_ms - b.latency_ms,
      render: (ms: number) => (
        <Typography.Text style={{ color: latencyColor(ms), fontFamily: 'monospace', fontWeight: 500 }}>
          {formatLatency(ms)}
        </Typography.Text>
      ),
    },
  ];

  return (
    <div className="jc-page">
      {/* Header */}
      <div style={{
        marginBottom: 20, display: 'flex', alignItems: 'center', gap: 12,
        borderBottom: '1px solid #1F2937', paddingBottom: 16, flexWrap: 'wrap',
      }}>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/accounts')} type="text" />
        <div style={{ flex: 1, minWidth: 240 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <Typography.Title level={4} style={{ margin: 0 }}>{accountDisplayName(account)}</Typography.Title>
            {account.is_default && <Tag color="blue">默认</Tag>}
          </div>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {account.user_id} · 创建于 {account.created_at?.slice(0, 10) || '-'}
          </Typography.Text>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4, flexWrap: 'wrap' }}>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>Token:</Typography.Text>
            <Typography.Text code copyable style={{ fontSize: 11 }}>
              {account.api_token}
            </Typography.Text>
          </div>
          <div style={{ marginTop: 6 }}>
            {activeSessions > 0 ? (
              <Badge status="processing" color="#22C55E" text={
                <Typography.Text style={{ fontSize: 12, color: '#22C55E' }}>
                  {activeSessions} 个活跃会话
                </Typography.Text>
              } />
            ) : (
              <Badge status="default" text={
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>无活跃会话</Typography.Text>
              } />
            )}
          </div>
        </div>
        <Space wrap>
          <Tooltip title="当客户端请求的模型不可用或未指定时，优先使用此账号的默认模型。模型列表来自该账号当前可用的 JoyCode 模型。">
            <QuestionCircleOutlined style={{ color: '#94A3B8', cursor: 'help' }} />
          </Tooltip>
          <Select
            style={{ width: 220 }}
            value={account.default_model || undefined}
            placeholder="默认模型"
            options={allModelOptions}
            allowClear
            loading={modelLoading}
            onChange={handleModelChange}
            disabled={savingModel}
            size="small"
          />
          {isClaudeModel(account.default_model) && (
            <Tooltip title="Claude 模型需要本机登录 JoyCode IDE">
              <InfoCircleOutlined style={{ color: '#F59E0B' }} />
            </Tooltip>
          )}
          <Button size="small" onClick={async () => {
            try {
              await api.renewToken(decodedKey);
              message.success('API Token 已更新');
              fetchData();
            } catch (e: unknown) {
              message.error(e instanceof Error ? e.message : '更新失败');
            }
          }}>
            重置 Token
          </Button>
          <Button size="small" icon={<ReloadOutlined />} onClick={() => { fetchData(); fetchModels(); fetchLogs(); }}>
            刷新
          </Button>
          <Popconfirm
            title={`确定要删除账号「${accountDisplayName(account)}」吗？`}
            description="删除后使用该密钥的客户端将无法访问"
            onConfirm={async () => {
              try {
                await api.removeAccount(decodedKey);
                message.success(`账号「${accountDisplayName(account)}」已删除`);
                navigate('/accounts');
              } catch (e: unknown) {
                message.error(e instanceof Error ? e.message : '删除账号失败');
              }
            }}
          >
            <Button size="small" danger icon={<DeleteOutlined />}>删除</Button>
          </Popconfirm>
        </Space>
      </div>

      {/* Live session status */}
      <Card
        size="small"
        style={{ marginBottom: 16, background: activeSessions > 0 ? 'rgba(34, 197, 94, 0.06)' : undefined, borderColor: activeSessions > 0 ? 'rgba(34, 197, 94, 0.3)' : undefined }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <Badge status={activeSessions > 0 ? 'processing' : 'default'} />
          <Typography.Text strong style={{ fontSize: 13 }}>
            实时状态
          </Typography.Text>
          <Typography.Text style={{ fontSize: 13 }}>
            当前有 <Typography.Text strong style={{ fontSize: 16, color: activeSessions > 0 ? '#22C55E' : undefined }}>{activeSessions}</Typography.Text> 个活跃连接
          </Typography.Text>
          {activeSessions > 0 && (
            <Tag color="blue">请求处理中</Tag>
          )}
          <div style={{ marginLeft: 'auto' }}>
            <Tooltip title="开启后每 5 秒自动刷新实时状态">
              <Segmented
                size="small"
                value={activeSessionsAutoRefresh ? 'on' : 'off'}
                onChange={(v) => setActiveSessionsAutoRefresh(v === 'on')}
                options={[
                  { label: '自动刷新 开', value: 'on' },
                  { label: '自动刷新 关', value: 'off' },
                ]}
              />
            </Tooltip>
          </div>
        </div>
      </Card>

      {/* Stats collapse header */}
      {stats && (
        <Card
          size="small"
          style={{ marginBottom: statsCollapsed ? 20 : 16, cursor: 'pointer' }}
          onClick={handleStatsToggle}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
            <span className="jc-section-title" style={{ margin: 0 }}>
              <FireOutlined />统计信息
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>今日 Token:</Typography.Text>
              <Typography.Text strong style={{ fontSize: 15, color: CHART_COLORS.secondary }}>
                {fmtTokens((statsSummary ?? stats).total_input_tokens + (statsSummary ?? stats).total_output_tokens)}
              </Typography.Text>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>累计 Token:</Typography.Text>
              <Typography.Text strong style={{ fontSize: 15 }}>
                {fmtTokens(((statsSummary ?? stats).all_time?.total_input_tokens ?? 0) + ((statsSummary ?? stats).all_time?.total_output_tokens ?? 0))}
              </Typography.Text>
            </div>
            <div onClick={(event) => event.stopPropagation()}>
              <Tooltip title="开启后每 5 秒自动刷新今日 Token 与累计 Token，展开详情时刷新完整统计">
                <Segmented
                  size="small"
                  value={statsAutoRefresh ? 'on' : 'off'}
                  onChange={(v) => setStatsAutoRefresh(v === 'on')}
                  options={[
                    { label: '自动刷新 开', value: 'on' },
                    { label: '自动刷新 关', value: 'off' },
                  ]}
                />
              </Tooltip>
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 4 }}>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                {statsCollapsed ? '展开详情' : '收起详情'}
              </Typography.Text>
              {statsCollapsed ? <DownOutlined style={{ fontSize: 12, color: '#94A3B8' }} /> : <UpOutlined style={{ fontSize: 12, color: '#94A3B8' }} />}
            </div>
          </div>
        </Card>
      )}

      {/* Stats panels */}
      {stats && !statsCollapsed && (
        <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
          <Col xs={24} md={12}>
            <Card
              size="small"
              style={{ height: '100%' }}
              title={<span className="jc-section-title"><ApiOutlined />请求统计</span>}
            >
              <Row gutter={[8, 12]}>
                <Col span={12}>
                  <Statistic title="今日请求" value={stats.total_requests} valueStyle={{ fontSize: 20, color: CHART_COLORS.primary }} prefix={<ApiOutlined />} />
                </Col>
                <Col span={12}>
                  <Statistic title="累计请求" value={stats.all_time?.total_requests ?? 0} valueStyle={{ fontSize: 20 }} />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="今日成功"
                    value={stats.success_count}
                    prefix={<CheckCircleOutlined />}
                    valueStyle={{ fontSize: 18, color: CHART_COLORS.primary }}
                  />
                  <Typography.Text type="secondary" style={{ fontSize: 11 }}>
                    占比 {stats.total_requests > 0 ? Math.round((stats.success_count / stats.total_requests) * 100) : 100}%
                  </Typography.Text>
                </Col>
                <Col span={12}>
                  <Statistic
                    title="今日失败"
                    value={stats.error_count}
                    prefix={<CloseCircleOutlined />}
                    valueStyle={{ fontSize: 18, color: stats.error_count > 0 ? CHART_COLORS.danger : CHART_COLORS.primary }}
                  />
                  <Typography.Text type="secondary" style={{ fontSize: 11 }}>
                    占比 {stats.total_requests > 0 ? Math.round((stats.error_count / stats.total_requests) * 100) : 0}%
                  </Typography.Text>
                </Col>
                <Col span={24}>
                  <Divider style={{ margin: '4px 0 8px' }} />
                  <Row gutter={8}>
                    <Col span={12}>
                      <Statistic
                        title="流式请求"
                        value={stats.stream_count}
                        prefix={<SwapOutlined />}
                        valueStyle={{ fontSize: 16 }}
                      />
                      <Tag color="blue" style={{ marginTop: 2 }}>
                        {stats.total_requests > 0 ? Math.round((stats.stream_count / stats.total_requests) * 100) : 0}%
                      </Tag>
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="平均延迟"
                        value={Math.round(stats.avg_latency_ms)}
                        suffix="ms"
                        prefix={<ThunderboltOutlined />}
                        valueStyle={{ fontSize: 16, color: stats.avg_latency_ms < 500 ? CHART_COLORS.primary : stats.avg_latency_ms < 1500 ? CHART_COLORS.warning : CHART_COLORS.danger }}
                      />
                    </Col>
                  </Row>
                </Col>
              </Row>
            </Card>
          </Col>

          <Col xs={24} md={12}>
            <Card
              size="small"
              style={{ height: '100%' }}
              title={<span className="jc-section-title"><FireOutlined />Token 消费</span>}
            >
              <Row gutter={[8, 12]}>
                <Col span={12}>
                  <Statistic
                    title="今日 Token"
                    value={fmtTokens(stats.total_input_tokens + stats.total_output_tokens)}
                    valueStyle={{ fontSize: 20, color: CHART_COLORS.secondary }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="累计 Token"
                    value={fmtTokens((stats.all_time?.total_input_tokens ?? 0) + (stats.all_time?.total_output_tokens ?? 0))}
                    valueStyle={{ fontSize: 20 }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic title="今日输入" value={fmtTokens(stats.total_input_tokens)} valueStyle={{ fontSize: 16 }} />
                </Col>
                <Col span={12}>
                  <Statistic title="今日输出" value={fmtTokens(stats.total_output_tokens)} valueStyle={{ fontSize: 16 }} />
                </Col>
                <Col span={24}>
                  <Divider style={{ margin: '4px 0 8px' }} />
                  <Row gutter={8}>
                    <Col span={12}>
                      <Statistic
                        title="平均每请求"
                        value={stats.total_requests > 0 ? fmtTokens(Math.round((stats.total_input_tokens + stats.total_output_tokens) / stats.total_requests)) : '-'}
                        suffix={stats.total_requests > 0 ? 'tokens' : ''}
                        valueStyle={{ fontSize: 15 }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="输入/输出比"
                        value={stats.total_output_tokens > 0 ? (stats.total_input_tokens / stats.total_output_tokens).toFixed(1) : '-'}
                        suffix={stats.total_output_tokens > 0 ? ':1' : ''}
                        valueStyle={{ fontSize: 15 }}
                      />
                    </Col>
                  </Row>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      )}

      {/* Hourly trend charts */}
      {stats && !statsCollapsed && stats.hourly && stats.hourly.length > 0 && (() => {
        // Build hourly chart with date-aware keys to avoid cross-day merging
        const hMap = new Map<string, { count: number; input_tokens: number; output_tokens: number; errors: number }>();
        for (const h of stats.hourly) {
          hMap.set(h.hour, h);
        }
        const now = new Date();
        const hourlyChartData: { label: string; count: number; input_tokens: number; output_tokens: number; errors: number }[] = [];
        for (let i = 23; i >= 0; i--) {
          const d = new Date(now.getTime() - i * 3600000);
          const key = `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}`;
          const entry = hMap.get(key);
          hourlyChartData.push({
            label: `${String(d.getHours()).padStart(2, '0')}:00`,
            count: entry?.count ?? 0,
            input_tokens: entry?.input_tokens ?? 0,
            output_tokens: entry?.output_tokens ?? 0,
            errors: entry?.errors ?? 0,
          });
        }
        return (
        <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
          <Col xs={24} lg={12}>
            <Card size="small" title="24 小时请求趋势">
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={hourlyChartData} margin={{ left: -10 }}>
                  <defs>
                    <linearGradient id="gradDetailReq" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART_COLORS.primary} stopOpacity={0.4} />
                      <stop offset="100%" stopColor={CHART_COLORS.primary} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gradDetailErr" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART_COLORS.danger} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={CHART_COLORS.danger} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="label" tick={{ fontSize: 11, fill: CHART_COLORS.axis }} interval={2} stroke={CHART_COLORS.grid} />
                  <YAxis tick={{ fontSize: 11, fill: CHART_COLORS.axis }} stroke={CHART_COLORS.grid} />
                  <RTooltip contentStyle={{ background: '#0E1223', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }} labelStyle={{ color: '#94A3B8' }} itemStyle={{ color: '#F8FAFC' }} />
                  <Area type="monotone" dataKey="count" name="请求数" stroke={CHART_COLORS.primary} fill="url(#gradDetailReq)" strokeWidth={2} />
                  <Area type="monotone" dataKey="errors" name="错误数" stroke={CHART_COLORS.danger} fill="url(#gradDetailErr)" strokeWidth={1.5} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </Col>
          <Col xs={24} lg={12}>
            <Card size="small" title="24 小时 Token 消耗趋势">
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={hourlyChartData} margin={{ left: -10 }}>
                  <defs>
                    <linearGradient id="gradDetailIn" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART_COLORS.secondary} stopOpacity={0.35} />
                      <stop offset="100%" stopColor={CHART_COLORS.secondary} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gradDetailOut" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART_COLORS.primary} stopOpacity={0.35} />
                      <stop offset="100%" stopColor={CHART_COLORS.primary} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="label" tick={{ fontSize: 11, fill: CHART_COLORS.axis }} interval={2} stroke={CHART_COLORS.grid} />
                  <YAxis tick={{ fontSize: 11, fill: CHART_COLORS.axis }} stroke={CHART_COLORS.grid} />
                  <RTooltip contentStyle={{ background: '#0E1223', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }} labelStyle={{ color: '#94A3B8' }} itemStyle={{ color: '#F8FAFC' }} />
                  <Area type="monotone" dataKey="input_tokens" name="输入 Token" stroke={CHART_COLORS.secondary} fill="url(#gradDetailIn)" strokeWidth={2} />
                  <Area type="monotone" dataKey="output_tokens" name="输出 Token" stroke={CHART_COLORS.primary} fill="url(#gradDetailOut)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>
        );
      })()}

      {/* Charts row */}
      {stats && !statsCollapsed && (stats.by_model.length > 0 || endpointData.length > 0) && (
        <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
          {stats.by_model.length > 0 && (
            <Col xs={24} lg={14}>
              <Card size="small" title={<span className="jc-section-title"><FireOutlined />模型使用分布</span>}>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={stats.by_model} layout="vertical" margin={{ left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                    <XAxis type="number" tick={{ fill: CHART_COLORS.axis }} stroke={CHART_COLORS.grid} />
                    <YAxis dataKey="model" type="category" width={100} tick={{ fontSize: 11, fill: CHART_COLORS.axis }} stroke={CHART_COLORS.grid} />
                    <RTooltip contentStyle={{ background: '#0E1223', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }} labelStyle={{ color: '#94A3B8' }} itemStyle={{ color: '#F8FAFC' }} />
                    <Bar dataKey="count" name="请求数" fill={CHART_COLORS.primary} radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          )}
          {endpointData.length > 0 && (
            <Col xs={24} lg={10}>
              <Card size="small" title={<span className="jc-section-title"><GlobalOutlined />端点调用分布</span>}>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={endpointData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={70}
                      label={({ name, percent }: any) => `${name || ''} ${((percent || 0) * 100).toFixed(0)}%`}
                      labelLine={{ stroke: '#475569', strokeWidth: 1 }}
                    >
                      {endpointData.map((_, i) => (
                        <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <RTooltip contentStyle={{ background: '#0E1223', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }} labelStyle={{ color: '#94A3B8' }} itemStyle={{ color: '#F8FAFC' }} />
                  </PieChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          )}
        </Row>
      )}

      {/* Request logs */}
      <Card
        size="small"
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <ClockCircleOutlined style={{ color: '#22C55E' }} />
            <span>请求日志</span>
            <Tag>{logTotal} 条</Tag>
          </div>
        }
      >
        <div className="jc-log-toolbar">
          <Input.Search
            className="jc-log-search"
            value={logSearchInput}
            placeholder="搜索模型、端点或错误信息"
            allowClear
            maxLength={200}
            enterButton="搜索"
            onChange={(event) => {
              const value = event.target.value;
              setLogSearchInput(value);
              if (!value && logSearch) {
                setLogPage(1);
                setLogSearch('');
              }
            }}
            onSearch={handleLogSearch}
          />
          <Segmented
            size="small"
            value={logFilter}
            onChange={(v) => { setLogFilter(v as 'all' | 'stream' | 'error'); setLogPage(1); }}
            options={[
              { label: '全部', value: 'all' },
              { label: '流式', value: 'stream' },
              { label: '错误', value: 'error' },
            ]}
          />
          <Tooltip title={logPage > 1 ? '自动刷新仅作用于第 1 页，查看历史页时不会打扰' : '开启后每 5 秒自动获取最新请求日志'}>
            <Segmented
              size="small"
              value={logAutoRefresh ? 'on' : 'off'}
              onChange={(v) => setLogAutoRefresh(v === 'on')}
              options={[
                { label: '自动刷新 开', value: 'on' },
                { label: '自动刷新 关', value: 'off' },
              ]}
            />
          </Tooltip>
        </div>
        <Table
          dataSource={logs}
          columns={logColumns}
          loading={logLoading}
          rowKey="id"
          size="small"
          pagination={{
            current: logPage, pageSize: logPageSize, total: logTotal,
            showSizeChanger: true, pageSizeOptions: [10, 20, 50, 100],
            showTotal: (t) => `共 ${t} 条`,
            onChange: (page, size) => {
              setLogPage(size === logPageSize ? page : 1);
              setLogPageSize(size);
            },
          }}
          scroll={{ x: 980 }}
          locale={{ emptyText: logSearch || logFilter !== 'all' ? '没有匹配的请求日志' : '暂无请求记录' }}
          expandable={{
            expandedRowRender: (record) => (
              <div style={{ padding: '8px 0 8px 12px' }}>
                <div className="jc-log-detail-header">
                  <Typography.Text strong>日志详情</Typography.Text>
                  <Tooltip title="复制完整请求详情">
                    <Button size="small" icon={<CopyOutlined />} onClick={() => handleCopyLog(record)}>
                      复制详情
                    </Button>
                  </Tooltip>
                </div>
                {record.status_code >= 400 && (
                  <div style={{
                    marginBottom: 10,
                    padding: '10px 12px',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    borderRadius: 6,
                    background: 'rgba(239, 68, 68, 0.08)',
                  }}>
                    <Typography.Text strong style={{ display: 'block', marginBottom: 6, color: CHART_COLORS.danger }}>
                      错误详情
                    </Typography.Text>
                    <pre style={{
                      margin: 0,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      fontSize: 12,
                      lineHeight: 1.6,
                      color: CHART_COLORS.danger,
                      fontFamily: "'Fira Code', monospace",
                    }}>
                      {record.error_message || `HTTP ${record.status_code}`}
                    </pre>
                  </div>
                )}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '120px minmax(0, 1fr)',
                  gap: '6px 12px',
                  fontSize: 12,
                }}>
                  {getLogDetailFields(record).map(({ label, value }) => (
                    <React.Fragment key={label}>
                      <Typography.Text type="secondary">{label}</Typography.Text>
                      <Typography.Text code={label === '请求 ID' || label === '端点'}>{value}</Typography.Text>
                    </React.Fragment>
                  ))}
                </div>
              </div>
            ),
          }}
        />
      </Card>
    </div>
  );
};

export default AccountDetail;
