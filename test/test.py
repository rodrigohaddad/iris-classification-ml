import unittest


class UserInfoTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_total_same_name_per_country(self):
        response = self.app.get('/total_same_name_per_country')
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data, Mock.response_total_same_name_per_country())


if __name__ == '__main__':
    unittest.main()
