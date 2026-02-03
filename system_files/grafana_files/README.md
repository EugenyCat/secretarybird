# secretarybird

### Connecting Grafana to ClickHouse

1. Open Grafana at [http://localhost:3000](http://localhost:3000) and log in.
2. Go to **Configuration > Data Sources**.
3. Click **Add data source** and select **Altinity plugin for ClickHouse** from the list of available data sources.

#### Connection configuration:

* **URL**: Specify the ClickHouse URL (for example, `http://<clickhouse-container-name>:8123`).
* **Default Database**: Enter the name of the ClickHouse database.
* **User and Password**: Add ClickHouse credentials if required.

#### Verification:

* Click **Save & Test** to verify. If the connection is established successfully, the data source will be ready for use.

---

### How to obtain an API key via a service account

1. **Log in to Grafana**:
   Go to [http://localhost:3000](http://localhost:3000) and log in.
2. **Go to administration**:
   In the left sidebar menu, click **Home** (or the name that corresponds to the main page) and select **Administration**.
3. **Create a service account**:
   Go to the **Users and access** or **Service accounts** section.
   Click the **Create service account** button.
4. **Service account settings**:
   Specify a name for the account (for example, *Backup Script*).
   Set the required permissions for this account (for example, **Admin** or **Editor** access level) to be able to interact with the API and retrieve dashboard information.
5. **Saving the key**:
   After creating the account, you will have access to the key. Be sure to copy it and store it securely, as you will not be able to view it again after this step.
