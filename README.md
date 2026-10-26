# MediSync AI — first runnable MVP

MediSync AI is a local, end-to-end prototype for medicine inventory visibility, baseline demand forecasting, shortage flags, expiry review, redistribution suggestions, reorder planning, CSV inventory import, and scenario simulation. It uses FastAPI, SQLAlchemy, pandas, NumPy, scikit-learn, and a responsive browser dashboard served by the API.

**All seeded records are synthetic demonstration data. Forecasts are exploratory baselines, not trained/validated production models. Redistribution and reorder results are decision support only and require human review.** No real hospital data or clinical outcomes are included.

## Windows quick start (SQLite demo)

1. Install Python 3.11 or newer. Open PowerShell in this `MediSync-AI` folder.
2. Create a virtual environment and install the packages:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that same window, then activate again.

3. Start the application:

   ```powershell
   uvicorn app.main:app --reload
   ```

4. Open <http://127.0.0.1:8000>. API documentation is at <http://127.0.0.1:8000/docs>. Stop with Ctrl+C.

The default SQLite database is created as `medisync.db` in the project folder and seeded on its first start. To reset the synthetic demo, stop the server and delete that file, then start the server again.

## Optional MySQL 8.0 Workbench setup

The app connects to the MySQL **server**, not to Workbench itself. In Workbench, connect to the local MySQL 8.0 server and run:

```sql
CREATE DATABASE medisync_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Copy `.env.example` to `.env`, set `DATABASE_URL` to your server credentials, and restart the app. The app loads `.env` automatically:

```text
DATABASE_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/medisync_ai?charset=utf8mb4
```

Alternatively set the variable for the current PowerShell session before starting:

```powershell
$env:DATABASE_URL = "mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/medisync_ai?charset=utf8mb4"
uvicorn app.main:app --reload
```

Use the same account/password that works in Workbench. If `mysql --version` is not recognized, that does not prevent this app from connecting; the MySQL Server service must be installed and running. Workbench is only the SQL client. Do not commit a real `.env` file or share its password.

## Sign-in and roles

The dashboard has four roles with server-side API permissions. On a fresh database, the local demo creates these accounts:

| Role | Username | Initial password |
|---|---|---|
| Admin | admin | Admin@123 |
| Hospital Manager | manager | Manager@123 |
| Pharmacist | pharmacist | Pharma@123 |
| Auditor (read only) | auditor | Audit@123 |

Sign in through the website. Admins can create additional accounts from Team access. Passwords are stored as salted PBKDF2 hashes. Demo login tokens are held in memory and are invalidated when the server restarts. This is a classroom MVP, not production authentication; change the demo passwords before entering non-synthetic information.

Roles are network-wide in this MVP: there is no assignment restricting a manager or pharmacist to one hospital.

## Features included

- Multi-hospital, multi-medicine inventory with batch and expiry dates; add batches and new hospitals directly through the dashboard form or by CSV.
- Record stock received, stock issued, or a corrected count for a batch; each movement stores before/after quantities and an audit record.
- 120 days of seeded synthetic daily consumption per hospital/medicine pair.
- 30-day demand baseline: scikit-learn linear trend over daily consumption, with non-negative output; average fallback for short histories. This is not a validated or accuracy-claimed model.
- Heuristic risk bands based on estimated days of cover and reorder level.
- Expiry date visibility and due-within-30-days KPI; FEFO ordering when selecting donor candidates.
- Redistribution candidates only when a donor retains stock above both its reorder buffer and 30-day forecast. Request, approve, reject, and complete transfers in the dashboard; completion adjusts both hospitals and preserves the source expiry.
- Simple reorder quantity suggestion, CSV inventory batch import, and non-mutating what-if simulation.
- SQLite local mode, MySQL 8.0 URL support, audit entries, and JSON API.
- Sign-in, Admin-created accounts, and API-enforced Admin, Hospital Manager, Pharmacist, and read-only Auditor permissions.
- Dashboard with inventory filters, prioritised risks, redistribution list, simulation modal, CSV upload, and recent activity.
- Team access screen for Admin account creation; API permissions protect routes even if dashboard controls are bypassed.

The MVP frontend is a polished responsive HTML/CSS/JavaScript dashboard served directly by FastAPI, so it runs without Node or a separate build step. The JSON API is already separated under `/api`; a React frontend can replace `app/static/index.html` later and call those same routes (configure the React dev server to proxy `/api` to `127.0.0.1:8000`).

## API endpoints

- `GET /api/health` — service/database mode and synthetic-data flag.
- `GET /api/dashboard` — summary KPIs, stock, redistribution, audit activity.
- `GET /api/inventory` — inventory, forecast, risk, expiry, and reorder fields.
- `POST /api/inventory` — add a batch; create a hospital or medicine when its name is new.
- `POST /api/inventory/{id}/movement` — record received, issued, or corrected batch stock with a reason.
- `GET /api/stock-movements` — recent batch quantity changes.
- `GET /api/transfers` — transfer request queue and status.
- `POST /api/transfers` — request a safe candidate transfer.
- `POST /api/transfers/{id}/approve`, `/reject`, `/complete` — review and execute a transfer; completing adjusts inventory.
- `GET /api/redistribution` — advisory transfer candidates.
- `GET /api/audit` — recent audit entries.
- `POST /api/import/csv` — multipart CSV import.
- `POST /api/simulate` — JSON scenario; does not alter stock.
- `GET /docs` — interactive API documentation.

CSV header (case-insensitive): `hospital,city,medicine,category,quantity,reorder_level,expiry_date`. Dates should be ISO `YYYY-MM-DD`; quantities must be non-negative. A sample file is included as `sample_inventory.csv`.

## Planned next steps (not implemented)

Hospital-specific account scoping, password reset/deactivation interface, durable expiring sessions, validated real-data ingestion, calibrated forecasting and backtesting, confidence intervals, supplier lead times, procurement integrations, alert delivery, regulatory/privacy review, deployment hardening, and broad automated test coverage. The current CSV import adds batches; it does not import historical consumption.

## Known MVP limits

Synthetic data is generated in-app and must never be presented as a real hospital dataset. Each demo combination has one synthetic batch; batch-level FEFO selection becomes more meaningful once multiple real batches are loaded. The forecast uses a simple trend model and the fixed 30-day horizon for risk/redistribution. It does not account for lead times, stock reservations, batch quality, transport constraints, or clinical criticality. This is a classroom prototype, not a production medical supply system.
