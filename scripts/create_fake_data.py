import json
import random

def create_fake_customers():
    staedte = ["Hamburg", "Berlin", "Munich", "Cologne", "Frankfurt"]
    strassen = ["Main Street", "Station Street", "Church Way", "Forest Street", "School Lane"]
    fahrzeugtypen = ["Eco Shuttle", "City Flex", "Premium Ride"]

    kunden = []
    for i in range(1, 21):
        kunde = {
            "kunden_id": i,
            "adresse": f"{random.choice(staedte)}, {random.choice(strassen)} {random.randint(1, 100)}",
            "marketing_permission": random.choice([True, False]),
            "rides_total": random.randint(0, 25),
            "lieblings_fahrzeugtyp": random.choice(fahrzeugtypen)
        }
        kunden.append(kunde)

    with open("customer_data.json", "w", encoding="utf-8") as f:
        json.dump(kunden, f, indent=2, ensure_ascii=False)
    print("Fake-data created: customer_data.json")

if __name__ == "__main__":
    create_fake_customers()
